"""Tests for the Vietnam screening project layer.

The transmission model cannot yet be calibrated for Vietnam — there are no targets and no
Vietnam parameter file — but the cost structures are fully determined by the supplied
spreadsheet and are tested here against it.
"""

import numpy
import pandas as pd
import pytest

from tbcore.economics import contract
from tbcore.economics.costs import compute_cost_streams, cost_items_from_spreadsheet
from projects.vietnam_screening import economics as vn_economics
from projects.vietnam_screening.config import AGE_GROUPS, VIETNAM_CONFIG
from projects.vietnam_screening.scenarios import (
    ALGORITHMS,
    PROVISIONAL_ALGORITHMS,
    all_scenarios,
    build_scenario,
)
from projects.vietnam_screening.targets import get_targets


@pytest.fixture(scope="module")
def items():
    return cost_items_from_spreadsheet("vnm")


# --------------------------------------------------------------------------------------
# Configuration and scenarios
# --------------------------------------------------------------------------------------


def test_national_scale_applies_no_population_scaling():
    assert VIETNAM_CONFIG.pop_scaling == 1.0
    assert VIETNAM_CONFIG.iso3 == "VNM"


def test_reachability_is_absent_per_specification():
    assert VIETNAM_CONFIG.stratifications == ()


def test_age_groups_retain_the_required_early_infection_breaks():
    assert {"0", "5", "15"}.issubset(set(AGE_GROUPS))


def test_economic_output_set_is_selected():
    assert VIETNAM_CONFIG.output_set == "economic"


def test_provisional_arms_excluded_by_default():
    """Placeholder sensitivities must not silently reach a results table."""
    default = {s.sc_id for s in all_scenarios()}
    assert not any(any(p in sc_id for p in PROVISIONAL_ALGORITHMS) for sc_id in default)
    full = {s.sc_id for s in all_scenarios(include_provisional=True)}
    assert len(full) == len(ALGORITHMS)


def test_scenario_excludes_young_children():
    scenario = build_scenario("acf_xpert")
    multipliers = scenario.scr_prgs[0].strata_coverage_multipliers["age"]
    assert multipliers["0"] == 0.0 and multipliers["5"] == 0.0


def test_unknown_algorithm_rejected():
    with pytest.raises(ValueError, match="Unknown algorithm"):
        build_scenario("acf_crystal_ball")


def test_targets_refuse_to_be_invented():
    with pytest.raises(NotImplementedError, match="No calibration targets"):
        get_targets()


# --------------------------------------------------------------------------------------
# Cost structures
# --------------------------------------------------------------------------------------


def test_every_algorithm_has_a_screening_and_diagnosis_cost(items):
    for algorithm in ALGORITHMS:
        structure = vn_economics.screening_cost_structure(algorithm, items)
        names = {c.name for c in structure.components}
        assert {"screening", "active_diagnosis"}.issubset(names)


def test_baseline_has_no_screening_cost(items):
    structure = vn_economics.baseline_cost_structure(items)
    assert {c.name for c in structure.components} == {
        "passive_diagnosis",
        "treatment_passive",
        "patient_cost_passive",
    }


def test_patient_costs_differ_by_detection_route(items):
    """The two-fold gap is a principal driver of cost-effectiveness."""
    structure = vn_economics.screening_cost_structure("acf_cxr_xpert", items)
    by_name = {c.name: c for c in structure.components}
    active = by_name["patient_cost_active"].unit_cost.mean
    passive = by_name["patient_cost_passive"].unit_cost.mean
    assert passive > 2 * active
    assert by_name["patient_cost_active"].quantity_output == contract.TREATMENT_STARTS_ACTIVE
    assert by_name["patient_cost_passive"].quantity_output == contract.TREATMENT_STARTS_PASSIVE


def test_routine_diagnosis_cost_applies_in_screening_arms_too(items):
    """Routine diagnosis continues alongside a campaign; omitting it understates costs."""
    structure = vn_economics.screening_cost_structure("acf_xpert", items)
    names = {c.name for c in structure.components}
    assert "passive_diagnosis" in names


def test_cxr_triage_is_cheaper_per_person_screened_than_universal_xpert(items):
    """Triage exists to avoid testing everyone with the expensive assay."""
    universal = vn_economics.screening_cost_structure("acf_xpert", items)
    triage = vn_economics.screening_cost_structure("acf_cxr_xpert", items)
    cost_of = lambda s: next(c for c in s.components if c.name == "screening").unit_cost.mean
    assert cost_of(triage) < cost_of(universal)


def test_missing_spreadsheet_row_names_what_is_available(items):
    truncated = {k: v for k, v in items.items() if k[0] != "Treatment, per person treated"}
    with pytest.raises(KeyError, match="Available rows"):
        vn_economics.baseline_cost_structure(truncated)


def test_cost_structure_consumes_contract_outputs(items):
    """Cost components must reference outputs the model actually produces."""
    algorithm = "acf_xpert"
    structure = vn_economics.screening_cost_structure(algorithm, items)

    synthetic = pd.DataFrame(
        {
            contract.TREATMENT_STARTS_PASSIVE: 1000.0,
            contract.TREATMENT_STARTS_ACTIVE: 400.0,
            f"n_screened_{algorithm}": 1_000_000.0,
            f"n_screen_positive_{algorithm}": 2000.0,
        },
        index=pd.Index(range(2027, 2032), name="year"),
    )
    table = compute_cost_streams(synthetic, structure)
    assert (table["total"] > 0).all()
    # Tolerance rather than exact equality: the two sides sum the same components in a
    # different order, so they can differ in the last bits.
    numpy.testing.assert_allclose(
        table["total"].values,
        (table["total_provider"] + table["total_patient"]).values,
        rtol=1e-12,
    )
    # Screening dominates provider cost in the campaign year, as a mass campaign should.
    assert table.loc[2027, "screening"] > table.loc[2027, "passive_diagnosis"]


def test_yld_weight_matches_the_spreadsheet_interval():
    draws = vn_economics.YLD_WEIGHT.sample(50_000, numpy.random.default_rng(0))
    assert draws.mean() == pytest.approx(0.333, rel=0.02)
    assert ((draws >= 0) & (draws <= 1)).all()
