"""End-to-end pipeline test: calibrate, run scenarios, store, reload.

Uses a deliberately tiny model and chain. The numbers are meaningless; what is being tested
is that the pipeline runs end to end and that a stored run can be reloaded with its
provenance intact — which is the requirement that posteriors and outputs are saved
automatically for later reuse.
"""

import pandas as pd
import pytest

from tbcore.calibration import (
    build_bcm,
    calibrate_and_run,
    difference_quantiles,
    draw_keys,
    iter_draws,
    normal_target,
    outputs_for_draw,
    run_scenarios,
)
from tbcore.economics import contract
from tbcore.config import AnalysisConfig, ModelConfig
from tbcore.interventions import ScreeningProgram, ScreeningTools, Scenario
from tbcore.parameters import get_parameters_and_priors
from tbcore.runs import (
    find_reusable_posterior,
    list_runs,
    load_run,
    new_run_id,
    save_run,
)

PROJECT = "pipeline_test"

FAST_ANALYSIS = AnalysisConfig(
    chains=2, cores=1, tune=10, draws=30, burn_in=10, full_runs_samples=5
)


@pytest.fixture(scope="module")
def tiny_config() -> ModelConfig:
    """Smallest model that still exercises every structural feature."""
    return ModelConfig(
        iso3="KIR",
        start_time=1900,
        end_time=2035,
        age_groups=("0", "5", "15"),
        pop_scaling=40483.0 / 119438.0,
        heterogeneous_mixing=False,
        output_set="economic",
    )


@pytest.fixture(scope="module")
def inputs():
    return get_parameters_and_priors("kir")


@pytest.fixture(scope="module")
def targets(inputs):
    return [
        normal_target("notifications", pd.Series([250.0], index=[2020]), tol_perc=40.0),
        normal_target("tb_prevalence_per100k", pd.Series([800.0], index=[2024])),
    ]


@pytest.fixture(autouse=True)
def isolated_output_root(tmp_path, monkeypatch):
    """Keep test runs out of the real output tree."""
    def fake_project_outputs(project: str):
        path = tmp_path / project
        path.mkdir(parents=True, exist_ok=True)
        return path

    monkeypatch.setattr("tbcore.runs.project_outputs", fake_project_outputs)


def screening_scenario() -> Scenario:
    return Scenario(
        sc_id="cxr_85",
        sc_name="CXR screening, 85%",
        scr_prgs=[
            ScreeningProgram(
                name="cxr_all",
                scr_tool=ScreeningTools.CXR,
                start_time=2026,
                end_time=2027,
                coverage_prop=0.85,
                strata_coverage_multipliers={"age": {"0": 0.0}},
            )
        ],
    )


# --------------------------------------------------------------------------------------
# Target validation
# --------------------------------------------------------------------------------------


def test_target_naming_a_missing_output_fails_before_sampling(tiny_config, inputs, targets):
    """A typo in a target name must not surface after an expensive calibration."""
    params, priors, tv_params = inputs
    bad = targets + [normal_target("not_an_output", pd.Series([1.0], index=[2020]))]
    with pytest.raises(KeyError, match="not_an_output"):
        build_bcm(tiny_config, params, priors, bad, tv_params)


def test_empty_target_data_rejected():
    with pytest.raises(ValueError, match="no data"):
        normal_target("notifications", pd.Series(dtype=float))


# --------------------------------------------------------------------------------------
# Calibration and scenarios
# --------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def pipeline_results(tiny_config, inputs, targets):
    params, priors, tv_params = inputs
    return calibrate_and_run(
        tiny_config,
        FAST_ANALYSIS,
        params,
        priors,
        targets,
        tv_params,
        scenarios=[screening_scenario()],
    )


def test_pipeline_produces_baseline_and_scenario(pipeline_results):
    assert set(pipeline_results["quantiles"]) == {"baseline", "cxr_85"}
    assert pipeline_results["timings"]["calibration_seconds"] >= 0


def test_quantiles_are_ordered(pipeline_results):
    quantiles = pipeline_results["quantiles"]["baseline"]
    median = quantiles["tb_incidence"]["0.5"]
    lower = quantiles["tb_incidence"]["0.025"]
    upper = quantiles["tb_incidence"]["0.975"]
    assert (lower <= median).all()
    assert (median <= upper).all()


def test_screening_averts_burden_in_every_draw(pipeline_results):
    """Paired differencing: the comparison is within a draw, not between summaries."""
    differences = pipeline_results["differences"]
    assert "cxr_85" in differences
    averted = differences["cxr_85"]["TB_averted"]
    assert (averted > 0).all()
    relative = differences["cxr_85"]["TB_averted_relative"]
    assert (relative.between(0, 1)).all()


def test_difference_rejects_unknown_reference(pipeline_results):
    with pytest.raises(KeyError, match="not among"):
        difference_quantiles(pipeline_results["full_runs"], reference="nonexistent")


# --------------------------------------------------------------------------------------
# Per-draw access, which is what the economic layer consumes
# --------------------------------------------------------------------------------------


def test_draw_keys_match_requested_sample_count(pipeline_results):
    results = pipeline_results["full_runs"]["baseline"].results
    assert len(draw_keys(results)) == FAST_ANALYSIS.full_runs_samples


def test_outputs_for_draw_are_time_indexed_single_level(pipeline_results):
    results = pipeline_results["full_runs"]["baseline"].results
    key, outputs = next(iter_draws(results))
    assert outputs.index.equals(results.index)
    assert outputs.columns.nlevels == 1
    assert "tb_incidence" in outputs.columns
    # The same draw must be retrievable directly.
    pd.testing.assert_frame_equal(outputs, outputs_for_draw(results, key))


def test_per_draw_outputs_satisfy_the_economics_contract(pipeline_results, tiny_config):
    """Costing consumes one draw at a time, so each draw must be a complete output set."""
    results = pipeline_results["full_runs"]["cxr_85"].results
    _, outputs = next(iter_draws(results))
    contract.validate_outputs(outputs, list(tiny_config.age_groups), ["cxr"])


def test_draws_are_paired_across_scenarios(pipeline_results):
    """Incremental results are only valid if scenarios share the same posterior draws."""
    baseline = draw_keys(pipeline_results["full_runs"]["baseline"].results)
    scenario = draw_keys(pipeline_results["full_runs"]["cxr_85"].results)
    assert baseline == scenario


def test_seeded_subsampling_is_reproducible(tiny_config, inputs, targets, pipeline_results):
    """Two runs with the same seed must select the same posterior draws."""
    params, priors, tv_params = inputs
    bcm = build_bcm(tiny_config, params, priors, targets, tv_params)
    first, _ = run_scenarios(
        {"baseline": bcm}, pipeline_results["idata"], FAST_ANALYSIS.burn_in, 5, random_seed=7
    )
    second, _ = run_scenarios(
        {"baseline": bcm}, pipeline_results["idata"], FAST_ANALYSIS.burn_in, 5, random_seed=7
    )
    assert draw_keys(first["baseline"].results) == draw_keys(second["baseline"].results)


def test_reserved_scenario_id_rejected(tiny_config, inputs, targets):
    params, priors, tv_params = inputs
    reserved = Scenario(sc_id="baseline", sc_name="clash")
    with pytest.raises(ValueError, match="reserved"):
        calibrate_and_run(
            tiny_config, FAST_ANALYSIS, params, priors, targets, tv_params, scenarios=[reserved]
        )


def test_posterior_reuse_skips_calibration(tiny_config, inputs, targets, pipeline_results):
    """Supplying a posterior must bypass sampling entirely."""
    params, priors, tv_params = inputs
    reused = calibrate_and_run(
        tiny_config,
        FAST_ANALYSIS,
        params,
        priors,
        targets,
        tv_params,
        scenarios=[screening_scenario()],
        idata=pipeline_results["idata"],
    )
    assert "calibration_seconds" not in reused["timings"]
    # Same posterior and same scenario must reproduce the same quantiles.
    pd.testing.assert_frame_equal(
        reused["quantiles"]["cxr_85"], pipeline_results["quantiles"]["cxr_85"]
    )


# --------------------------------------------------------------------------------------
# Run storage and provenance
# --------------------------------------------------------------------------------------


def test_saved_run_round_trips(tiny_config, pipeline_results):
    run_id = new_run_id("smoke")
    save_run(
        PROJECT,
        run_id,
        tiny_config,
        analysis_config=FAST_ANALYSIS,
        idata=pipeline_results["idata"],
        quantiles=pipeline_results["quantiles"],
        description="pipeline test",
        timings=pipeline_results["timings"],
    )

    stored = load_run(PROJECT, run_id)
    assert stored.manifest.config_hash == tiny_config.config_hash()
    assert stored.manifest.description == "pipeline test"
    assert stored.manifest.has_posterior
    assert sorted(stored.manifest.scenarios) == ["baseline", "cxr_85"]
    assert stored.config["model"]["iso3"] == "KIR"
    assert stored.config["analysis"]["draws"] == FAST_ANALYSIS.draws

    reloaded = stored.quantiles("cxr_85")
    assert not reloaded.empty
    assert stored.posterior() is not None


def test_run_records_git_provenance(tiny_config, pipeline_results):
    run_id = new_run_id("provenance")
    save_run(PROJECT, run_id, tiny_config, quantiles=pipeline_results["quantiles"])
    manifest = load_run(PROJECT, run_id).manifest
    assert manifest.git_commit
    assert isinstance(manifest.git_dirty, bool)
    assert manifest.created.startswith("20")


def test_missing_run_lists_available_ones(tiny_config, pipeline_results):
    save_run(PROJECT, new_run_id("present"), tiny_config, quantiles=pipeline_results["quantiles"])
    with pytest.raises(FileNotFoundError, match="present"):
        load_run(PROJECT, "does_not_exist")


def test_missing_scenario_lists_available_ones(tiny_config, pipeline_results):
    run_id = new_run_id("scenarios")
    save_run(PROJECT, run_id, tiny_config, quantiles=pipeline_results["quantiles"])
    with pytest.raises(FileNotFoundError, match="cxr_85"):
        load_run(PROJECT, run_id).quantiles("nonexistent")


def test_posterior_reuse_is_keyed_on_configuration(tiny_config, pipeline_results):
    """A posterior may only be reused by an identically configured model."""
    save_run(
        PROJECT,
        new_run_id("calibration"),
        tiny_config,
        idata=pipeline_results["idata"],
        quantiles=pipeline_results["quantiles"],
    )

    found = find_reusable_posterior(PROJECT, tiny_config)
    assert found is not None and found.config_hash == tiny_config.config_hash()

    # A different horizon is a different model, so the posterior must not be offered.
    assert find_reusable_posterior(PROJECT, tiny_config.with_changes(end_time=2046)) is None


def test_runs_are_listed_newest_first(tiny_config, pipeline_results):
    for name in ["first", "second", "third"]:
        save_run(PROJECT, new_run_id(name), tiny_config, quantiles=pipeline_results["quantiles"])
    listed = list_runs(PROJECT)
    assert len(listed) >= 3
    assert [m.run_id for m in listed] == sorted((m.run_id for m in listed), reverse=True)


def test_run_without_posterior_is_marked_as_such(tiny_config, pipeline_results):
    run_id = new_run_id("no_posterior")
    save_run(PROJECT, run_id, tiny_config, quantiles=pipeline_results["quantiles"])
    stored = load_run(PROJECT, run_id)
    assert not stored.manifest.has_posterior
    with pytest.raises(FileNotFoundError, match="no posterior"):
        stored.posterior()
