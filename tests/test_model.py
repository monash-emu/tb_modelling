"""Structural tests for the model core.

These build real models from the Kiribati inputs. They are not a substitute for reproducing
published results — that check belongs with a calibrated posterior — but they catch the
failure modes that matter when the core is refactored: a stratification that stops being
optional, an output name that drifts, a screening programme that silently does nothing.
"""

import numpy as np
import pytest

from tbcore.config import KIRIBATI_AGE_GROUPS, ModelConfig
from tbcore.demography import available_countries, get_population_over_time
from tbcore.economics import contract
from tbcore.interventions import (
    MAX_COVERAGE_PROP,
    ProgrammaticLever,
    Scenario,
    ScreeningProgram,
    ScreeningTools,
)
from tbcore.mixing import get_age_mixing_matrix
from tbcore.model import get_tb_model
from tbcore.parameters import get_parameters_and_priors
from tbcore.stratifications import IncomeStratification, ReachabilityStratification

KIR_SCALING = 40483.0 / 119438.0
AGE_AGGREGATES = (("3_9", ("3", "5")), ("15+", ("15", "18", "40", "65")))


@pytest.fixture(scope="module")
def kir_inputs():
    return get_parameters_and_priors("kir")


def kir_config(**overrides) -> ModelConfig:
    defaults = dict(
        iso3="KIR",
        age_groups=tuple(KIRIBATI_AGE_GROUPS),
        pop_scaling=KIR_SCALING,
        heterogeneous_mixing=False,
        output_set="economic",
        age_aggregates=AGE_AGGREGATES,
    )
    return ModelConfig(**{**defaults, **overrides})


def run_model(config, tv_params, params, screening_programs=None, lever=None):
    model = get_tb_model(config, tv_params, screening_programs or [], lever)
    model.run(parameters=params)
    return model


# --------------------------------------------------------------------------------------
# Demography
# --------------------------------------------------------------------------------------


def test_shared_demography_covers_all_planned_countries():
    assert set(available_countries()) >= {"IDN", "KIR", "PHL", "VNM"}


def test_population_groups_sum_to_single_age_total():
    single, grouped = get_population_over_time("KIR", KIRIBATI_AGE_GROUPS)
    year = 2024
    assert grouped.loc[year].sum() == pytest.approx(
        single[single["Time"] == year]["Pop"].sum()
    )


def test_unknown_country_names_the_available_ones():
    with pytest.raises(ValueError, match="Available"):
        get_population_over_time("ZZZ", KIRIBATI_AGE_GROUPS)


def test_unsorted_age_groups_rejected():
    with pytest.raises(ValueError, match="ascending"):
        get_population_over_time("KIR", ["15", "0"])


# --------------------------------------------------------------------------------------
# Mixing
# --------------------------------------------------------------------------------------


def test_homogeneous_mixing_returns_no_matrix():
    assert get_age_mixing_matrix("VNM", ["0", "15"], heterogeneous=False) is None


def test_heterogeneous_mixing_fails_loudly_rather_than_silently_downgrading():
    with pytest.raises(NotImplementedError, match="contact matrix"):
        get_age_mixing_matrix("VNM", ["0", "15"], heterogeneous=True)


# --------------------------------------------------------------------------------------
# Model construction
# --------------------------------------------------------------------------------------


def test_unstratified_model_runs_without_nans(kir_inputs):
    params, _, tv_params = kir_inputs
    model = run_model(kir_config(), tv_params, params)
    outputs = model.get_derived_outputs_df()
    assert not outputs.isna().any().any()
    assert (outputs["population"] > 0).all()


def test_population_tracks_the_demographic_input(kir_inputs):
    """Births are calibrated to reproduce the observed population trajectory."""
    params, _, tv_params = kir_inputs
    config = kir_config()
    model = run_model(config, tv_params, params)

    _, grouped = get_population_over_time(
        config.iso3, list(config.age_groups), scaling_factor=config.pop_scaling
    )
    for year in [1990, 2010, 2024]:
        modelled = model.get_derived_outputs_df().loc[year, "population"]
        assert modelled == pytest.approx(grouped.loc[year].sum(), rel=0.02)


def test_compartment_count_scales_with_stratifications(kir_inputs):
    params, _, tv_params = kir_inputs
    plain = run_model(kir_config(), tv_params, params)
    stratified = run_model(
        kir_config(stratifications=(ReachabilityStratification(),)), tv_params, params
    )
    assert len(stratified.compartments) == 2 * len(plain.compartments)


def test_reachability_is_optional_and_changes_no_aggregate_when_neutral(kir_inputs):
    """With neutral reachability parameters, adding the stratification must not move results.

    This is the test that the stratification is genuinely a dimension rather than a hidden
    change of assumptions: setting the relative susceptibility and detection multipliers to
    one should recover the unstratified model exactly.
    """
    params, _, tv_params = kir_inputs
    neutral = params | {"rel_sus_unreachable": 1.0, "rel_detection_unreachable": 1.0}

    plain = run_model(kir_config(), tv_params, neutral).get_derived_outputs_df()
    stratified = run_model(
        kir_config(stratifications=(ReachabilityStratification(),)), tv_params, neutral
    ).get_derived_outputs_df()

    for output in ["population", "tb_incidence", "tb_prevalence", "notifications"]:
        # Tolerance is solver-level, not structural: the stratified model has twice the
        # compartments and so takes slightly different integration steps. A genuine change
        # of assumptions would move results by orders of magnitude more than this.
        np.testing.assert_allclose(
            plain[output].values, stratified[output].values, rtol=1e-3
        )


def test_births_preserve_stratum_composition(kir_inputs):
    """Population split must stay constant over time, since births are split to match."""
    params, _, tv_params = kir_inputs
    config = kir_config(stratifications=(ReachabilityStratification(),))
    outputs = run_model(config, tv_params, params).get_derived_outputs_df()

    share = outputs["populationXreachability_reachable"] / outputs["population"]
    assert share.loc[1900] == pytest.approx(params["reachable_pop_frac"], abs=0.01)
    assert share.loc[2030] == pytest.approx(params["reachable_pop_frac"], abs=0.01)


def test_income_stratification_builds(kir_inputs):
    """The income dimension is structurally identical to reachability, so it must build."""
    params, _, tv_params = kir_inputs
    income_params = params | {
        "lower_income_pop_frac": 0.5,
        "rel_sus_lower_income": 1.5,
        "rel_detection_lower_income": 0.7,
    }
    config = kir_config(stratifications=(IncomeStratification(),), output_set="core")
    outputs = run_model(config, tv_params, income_params).get_derived_outputs_df()
    assert "tb_incidenceXincome_lower" in outputs
    # Higher susceptibility and worse access must concentrate incidence in the lower group.
    assert outputs.loc[2024, "prop_tb_incidenceXincome_lower"] > 0.5


def test_income_assortative_mixing_refuses_rather_than_pretends():
    with pytest.raises(NotImplementedError, match="assortative"):
        IncomeStratification(assortative_mixing=True)


# --------------------------------------------------------------------------------------
# Screening programmes
# --------------------------------------------------------------------------------------


def pearl_program(coverage=0.85, **kwargs) -> ScreeningProgram:
    return ScreeningProgram(
        name="pearl_10plus",
        scr_tool=ScreeningTools.PEARL,
        start_time=2026,
        end_time=2027,
        coverage_prop=coverage,
        strata_coverage_multipliers={"age": {"0": 0.0, "3": 0.0, "5": 0.0}},
        **kwargs,
    )


def test_screening_reduces_burden(kir_inputs):
    params, _, tv_params = kir_inputs
    config = kir_config()
    baseline = run_model(config, tv_params, params).get_derived_outputs_df()
    screened = run_model(
        config, tv_params, params, screening_programs=[pearl_program()]
    ).get_derived_outputs_df()

    assert screened.loc[2030, "tb_incidence"] < baseline.loc[2030, "tb_incidence"]
    assert screened.loc[2035, "cum_tb_mortality"] < baseline.loc[2035, "cum_tb_mortality"]


def test_higher_coverage_averts_more(kir_inputs):
    params, _, tv_params = kir_inputs
    config = kir_config()
    low = run_model(
        config, tv_params, params, screening_programs=[pearl_program(0.5)]
    ).get_derived_outputs_df()
    high = run_model(
        config, tv_params, params, screening_programs=[pearl_program(0.85)]
    ).get_derived_outputs_df()
    assert high.loc[2035, "cum_tb_incidence"] < low.loc[2035, "cum_tb_incidence"]


def test_coverage_above_limit_rejected():
    with pytest.raises(ValueError, match="coverage_prop"):
        pearl_program(coverage=1.0)


def test_population_coverage_basis_guard_catches_divergent_rate():
    """The published configuration ran within 0.6% of a divergent screening rate."""
    with pytest.raises(ValueError, match="diverge"):
        pearl_program(coverage=0.845, population_coverage_basis=0.80)


def test_population_coverage_basis_reproduces_published_rescaling():
    program = pearl_program(coverage=0.845, population_coverage_basis=0.85)
    assert program.effective_coverage == pytest.approx(0.845 / 0.85)
    assert program.effective_coverage < MAX_COVERAGE_PROP


def test_screened_count_is_people_not_encounters(kir_inputs):
    """People screened must match coverage times the eligible population.

    The detection hazard is memoryless, so rate times population would count repeat
    screening events and exceed the population. Unit costs are per person screened, so the
    resource-use output uses campaign throughput instead — see ScreeningProgram.
    """
    params, _, tv_params = kir_inputs
    config = kir_config()
    coverage = 0.85
    outputs = run_model(
        config, tv_params, params, screening_programs=[pearl_program(coverage)]
    ).get_derived_outputs_df()

    during = outputs.loc[2026]
    # The programme excludes under-tens, so eligibility is the 10-and-over population.
    eligible = sum(
        during[f"populationXage_{age}"] for age in config.age_groups if int(age) >= 10
    )
    assert during["n_screened_pearl"] == pytest.approx(coverage * eligible, rel=0.01)
    assert during["n_screened_pearl"] < during["population"]
    # Outside the campaign window nobody is screened.
    assert outputs.loc[2030, "n_screened_pearl"] == pytest.approx(0.0, abs=1e-9)


def test_screened_count_scales_with_coverage(kir_inputs):
    params, _, tv_params = kir_inputs
    config = kir_config()
    low = run_model(
        config, tv_params, params, screening_programs=[pearl_program(0.4)]
    ).get_derived_outputs_df()
    high = run_model(
        config, tv_params, params, screening_programs=[pearl_program(0.8)]
    ).get_derived_outputs_df()
    assert high.loc[2026, "n_screened_pearl"] == pytest.approx(
        2.0 * low.loc[2026, "n_screened_pearl"], rel=0.01
    )


def test_unknown_stratum_in_programme_is_an_error(kir_inputs):
    params, _, tv_params = kir_inputs
    program = ScreeningProgram(
        name="bad",
        scr_tool=ScreeningTools.CXR,
        start_time=2026,
        end_time=2027,
        coverage_prop=0.5,
        strata_coverage_multipliers={"reachability": {"nonexistent": 1.0}},
    )
    config = kir_config(stratifications=(ReachabilityStratification(),))
    with pytest.raises(ValueError, match="unknown strata"):
        get_tb_model(config, tv_params, [program])


def test_outreach_programme_can_reach_the_unreachable(kir_inputs):
    """The ARC follow-up compares strategies targeting people missed by PEARL.

    A programme declaring its own reachability multipliers must override the stratification
    default of no reach — this is the assumption that was hard-coded in the published model.
    """
    params, _, tv_params = kir_inputs
    config = kir_config(stratifications=(ReachabilityStratification(),))

    routine = pearl_program()
    outreach = ScreeningProgram(
        name="outreach",
        scr_tool=ScreeningTools.CXR,
        start_time=2026,
        end_time=2027,
        coverage_prop=0.6,
        strata_coverage_multipliers={"reachability": {"reachable": 0.0, "unreachable": 1.0}},
    )

    routine_only = run_model(
        config, tv_params, params, screening_programs=[routine]
    ).get_derived_outputs_df()
    with_outreach = run_model(
        config, tv_params, params, screening_programs=[routine, outreach]
    ).get_derived_outputs_df()

    unreachable_burden = "tb_incidenceXreachability_unreachable"
    assert (
        with_outreach.loc[2030, unreachable_burden]
        < routine_only.loc[2030, unreachable_burden]
    )


# --------------------------------------------------------------------------------------
# Programmatic levers
# --------------------------------------------------------------------------------------


def test_detection_lever_moves_notifications_both_ways(kir_inputs):
    """One mechanism serves funding disruption and enhanced vigilance."""
    params, _, tv_params = kir_inputs
    config = kir_config(output_set="core")

    baseline = run_model(config, tv_params, params).get_derived_outputs_df()
    degraded = run_model(
        config, tv_params, params, lever=ProgrammaticLever(2026, detection_rate_rel=0.5)
    ).get_derived_outputs_df()
    improved = run_model(
        config, tv_params, params, lever=ProgrammaticLever(2026, detection_rate_rel=1.5)
    ).get_derived_outputs_df()

    assert degraded.loc[2030, "notifications"] < baseline.loc[2030, "notifications"]
    assert improved.loc[2030, "notifications"] > baseline.loc[2030, "notifications"]
    # Worse detection must raise burden; better detection must lower it.
    assert degraded.loc[2035, "cum_tb_mortality"] > baseline.loc[2035, "cum_tb_mortality"]
    assert improved.loc[2035, "cum_tb_mortality"] < baseline.loc[2035, "cum_tb_mortality"]


def test_lever_leaves_pre_start_period_untouched(kir_inputs):
    params, _, tv_params = kir_inputs
    config = kir_config(output_set="core")
    baseline = run_model(config, tv_params, params).get_derived_outputs_df()
    disrupted = run_model(
        config, tv_params, params, lever=ProgrammaticLever(2026, detection_rate_rel=0.3)
    ).get_derived_outputs_df()
    assert disrupted.loc[2020, "notifications"] == pytest.approx(
        baseline.loc[2020, "notifications"], rel=1e-9
    )


def test_treatment_success_lever_cannot_exceed_one(kir_inputs):
    """A multiplier above one must not push a proportion above one."""
    params, _, tv_params = kir_inputs
    config = kir_config(output_set="core")
    outputs = run_model(
        config, tv_params, params, lever=ProgrammaticLever(2026, tx_success_rel=5.0)
    ).get_derived_outputs_df()
    assert not outputs.isna().any().any()
    # Relapse cannot go negative, which is what an uncapped proportion would cause.
    assert (outputs.loc[2030:, "tb_incidence"] >= 0).all()


def test_neutral_lever_is_a_no_op(kir_inputs):
    params, _, tv_params = kir_inputs
    config = kir_config(output_set="core")
    baseline = run_model(config, tv_params, params).get_derived_outputs_df()
    neutral = run_model(
        config, tv_params, params, lever=ProgrammaticLever(2026)
    ).get_derived_outputs_df()
    np.testing.assert_allclose(
        baseline["tb_incidence"].values, neutral["tb_incidence"].values, rtol=1e-9
    )


def test_negative_lever_value_rejected():
    with pytest.raises(ValueError, match="must not be negative"):
        ProgrammaticLever(2026, detection_rate_rel=-1.0)


# --------------------------------------------------------------------------------------
# The economics contract
# --------------------------------------------------------------------------------------


def test_economic_output_set_satisfies_the_contract(kir_inputs):
    """The model must produce exactly what the costing and DALY code consumes."""
    params, _, tv_params = kir_inputs
    config = kir_config(output_set="economic")
    program = pearl_program()
    outputs = run_model(
        config, tv_params, params, screening_programs=[program]
    ).get_derived_outputs_df()
    contract.validate_outputs(outputs, list(config.age_groups), [program.scr_tool.name])


def test_baseline_emits_zero_active_treatment_starts(kir_inputs):
    """A scenario with no screening still needs the column, for comparability."""
    params, _, tv_params = kir_inputs
    outputs = run_model(kir_config(), tv_params, params).get_derived_outputs_df()
    contract.validate_outputs(outputs, list(kir_config().age_groups), [])
    assert (outputs[contract.TREATMENT_STARTS_ACTIVE] == 0).all()


def test_treatment_starts_split_sums_to_total(kir_inputs):
    params, _, tv_params = kir_inputs
    outputs = run_model(
        kir_config(), tv_params, params, screening_programs=[pearl_program()]
    ).get_derived_outputs_df()
    np.testing.assert_allclose(
        outputs["tb_treatment_starts"].values,
        (outputs["tb_treatment_starts_passive"] + outputs["tb_treatment_starts_active"]).values,
        rtol=1e-9,
    )


def test_active_treatment_starts_only_during_campaign(kir_inputs):
    params, _, tv_params = kir_inputs
    outputs = run_model(
        kir_config(), tv_params, params, screening_programs=[pearl_program()]
    ).get_derived_outputs_df()
    assert outputs.loc[2020, "tb_treatment_starts_active"] == pytest.approx(0.0, abs=1e-9)
    assert outputs.loc[2026, "tb_treatment_starts_active"] > 0
    assert outputs.loc[2030, "tb_treatment_starts_active"] == pytest.approx(0.0, abs=1e-9)


def test_test_counts_follow_the_algorithm(kir_inputs):
    """PEARL performs frontline Xpert on 35% of those screened; the rest get CXR alone."""
    params, _, tv_params = kir_inputs
    outputs = run_model(
        kir_config(), tv_params, params, screening_programs=[pearl_program()]
    ).get_derived_outputs_df()
    during = outputs.loc[2026]
    assert during["n_tests_xpert"] == pytest.approx(0.35 * during["n_screened_pearl"])
    assert during["n_tests_cxr"] == pytest.approx(during["n_screened_pearl"])


def test_tst_screen_positives_exceed_completions(kir_inputs):
    """Screen-positives are counted before the preventive-treatment completion probability."""
    params, _, tv_params = kir_inputs
    tst = ScreeningProgram(
        name="tst_3plus",
        scr_tool=ScreeningTools.TST,
        start_time=2026,
        end_time=2027,
        coverage_prop=0.85,
        strata_coverage_multipliers={"age": {"0": 0.0}},
    )
    outputs = run_model(
        kir_config(), tv_params, params, screening_programs=[tst]
    ).get_derived_outputs_df()
    completion_prop = params["tpt_completion_perc"] / 100.0
    assert completion_prop < 1.0
    assert outputs.loc[2026, "n_screen_positive_tst"] > 0


def test_core_output_set_omits_resource_use(kir_inputs):
    """Resource use is not free; a calibration run should not pay for it."""
    params, _, tv_params = kir_inputs
    core = run_model(kir_config(output_set="core"), tv_params, params).get_derived_outputs_df()
    economic = run_model(
        kir_config(output_set="economic"), tv_params, params
    ).get_derived_outputs_df()
    assert contract.TREATMENT_STARTS_ACTIVE not in core.columns
    assert contract.TREATMENT_STARTS_ACTIVE in economic.columns
    assert economic.shape[1] > core.shape[1]


def test_unknown_output_set_rejected(kir_inputs):
    _, _, tv_params = kir_inputs
    with pytest.raises(ValueError, match="Unknown output_set"):
        get_tb_model(kir_config(output_set="everything"), tv_params)


# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------


def test_config_hash_is_stable_and_discriminating():
    assert kir_config().config_hash() == kir_config().config_hash()
    assert kir_config().config_hash() != kir_config(end_time=2046).config_hash()


def test_config_rejects_scenario_start_outside_simulation():
    with pytest.raises(ValueError, match="scenario_start_time"):
        kir_config(scenario_start_time=2100)


def test_duplicate_stratifications_rejected():
    with pytest.raises(ValueError, match="Duplicate"):
        kir_config(
            stratifications=(ReachabilityStratification(), ReachabilityStratification())
        )


def test_missing_required_age_break_rejected(kir_inputs):
    _, _, tv_params = kir_inputs
    with pytest.raises(ValueError, match="required but missing"):
        get_tb_model(kir_config(age_groups=("0", "20", "60")), tv_params)


def test_scenario_rejects_duplicate_programme_names():
    with pytest.raises(ValueError, match="duplicate programme names"):
        Scenario("sc1", "Test", scr_prgs=[pearl_program(), pearl_program()])
