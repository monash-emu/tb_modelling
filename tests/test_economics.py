"""Tests for the economic post-processing layer.

These run without building a model: the economics layer consumes derived-output dataframes,
so it can be tested against synthetic outputs. That separation is the point of the design —
cost assumptions can be changed and re-verified without touching the transmission model.
"""

import numpy as np
import pandas as pd
import pytest

from tbcore.economics import cea, contract, costs, dalys, discounting, inputs
from tbcore.economics.cea import (
    acceptability_curve,
    efficiency_frontier,
    net_monetary_benefit,
)
from tbcore.economics.costs import (
    CostStructure,
    compute_cost_streams,
    sample_unit_costs,
    total_costs,
)
from tbcore.economics.dalys import (
    age_weighted_daly_values,
    compute_daly_streams,
    total_dalys,
)
from tbcore.economics.discounting import discount_factors

ISO3 = "vnm"


# --------------------------------------------------------------------------------------
# Discounting
# --------------------------------------------------------------------------------------


def test_discount_factor_is_one_at_reference_year():
    factors = discount_factors([2026, 2027, 2028], rate=0.03, reference_year=2026)
    assert factors.loc[2026] == pytest.approx(1.0)
    assert factors.loc[2027] == pytest.approx(1 / 1.03)
    assert factors.loc[2028] == pytest.approx(1 / 1.03**2)


def test_zero_rate_leaves_values_unchanged():
    series = pd.Series([10.0, 20.0, 30.0], index=[2026, 2027, 2028])
    assert discounting.discount_series(series, rate=0.0).tolist() == series.tolist()


def test_discounted_total_respects_horizon():
    series = pd.Series(100.0, index=range(2020, 2051))
    total = discounting.discounted_total(
        series, rate=0.0, start_year=2026, end_year=2035
    )
    assert total == pytest.approx(1000.0)  # ten years at 100, undiscounted


def test_discounted_total_rejects_empty_horizon():
    series = pd.Series(1.0, index=range(2020, 2030))
    with pytest.raises(ValueError, match="No data within horizon"):
        discounting.discounted_total(series, start_year=2040, end_year=2050)


def test_negative_discount_rate_rejected():
    with pytest.raises(ValueError, match="must not be negative"):
        discount_factors([2026], rate=-0.01)


# --------------------------------------------------------------------------------------
# Input loading and sampling
# --------------------------------------------------------------------------------------


def test_age_daly_table_parses_all_bands():
    table = inputs.load_age_daly_table(ISO3)
    assert table.index.min() == 0
    assert table.index.max() == 85
    # Years of life lost must fall with age.
    assert table["yll"].is_monotonic_decreasing
    assert (table["post_tb_daly"] > 0).all()


def test_unit_costs_load_from_spreadsheet():
    items = costs.cost_items_from_spreadsheet(ISO3)
    assert len(items) >= 9
    assert all(q.mean > 0 and q.sd > 0 for q in items.values())
    # Patient costs are far higher for passively detected cases; this gap drives the CEA,
    # so assert it survives loading.
    patient = {k: v for k, v in items.items() if "patient cost" in k[0].lower()}
    assert len(patient) == 2
    acf = next(v.mean for k, v in patient.items() if "ACF" in k[1])
    no_acf = next(v.mean for k, v in patient.items() if "No screening" in k[1])
    assert no_acf > 2 * acf


def test_gamma_sampling_recovers_mean_and_sd():
    quantity = inputs.UncertainQuantity("test", mean=100.0, sd=20.0)
    draws = quantity.sample(200_000, np.random.default_rng(0))
    assert draws.mean() == pytest.approx(100.0, rel=0.01)
    assert draws.std() == pytest.approx(20.0, rel=0.02)
    assert (draws > 0).all()


def test_beta_sampling_respects_bounds():
    sd = inputs.sd_from_ci(0.224, 0.454)
    quantity = inputs.UncertainQuantity("yld_weight", mean=0.333, sd=sd, kind="beta")
    draws = quantity.sample(100_000, np.random.default_rng(1))
    assert draws.mean() == pytest.approx(0.333, rel=0.01)
    assert ((draws >= 0) & (draws <= 1)).all()


def test_zero_sd_is_deterministic():
    quantity = inputs.UncertainQuantity("fixed", mean=5.0, sd=0.0)
    assert set(quantity.sample(10, np.random.default_rng(0))) == {5.0}


# --------------------------------------------------------------------------------------
# Age remapping of DALY values
# --------------------------------------------------------------------------------------


def _uniform_single_age_pop(year=2026, max_age=99, per_age=1000.0):
    return pd.DataFrame(
        {"Time": year, "Age": range(max_age + 1), "Pop": per_age}
    )


def test_age_weighted_values_match_band_value_when_group_matches_band():
    """A model group spanning exactly one source band must take that band's value."""
    table = pd.DataFrame(
        {"yll": [30.0, 20.0, 10.0], "post_tb_daly": [5.0, 3.0, 1.0]},
        index=pd.Index([0, 50, 80], name="age_lower"),
    )
    values = age_weighted_daly_values(
        table, _uniform_single_age_pop(), ["0", "50", "80"], year=2026, max_age=99
    )
    assert values.loc["0", "yll"] == pytest.approx(30.0)
    assert values.loc["50", "yll"] == pytest.approx(20.0)
    assert values.loc["80", "yll"] == pytest.approx(10.0)


def test_coarse_group_takes_population_weighted_mean():
    """A model group spanning two source bands must average them by population."""
    table = pd.DataFrame(
        {"yll": [30.0, 10.0], "post_tb_daly": [4.0, 2.0]},
        index=pd.Index([0, 50], name="age_lower"),
    )
    # 0-49 has 50 single ages, 50-99 has 50, so a single 0+ group averages evenly.
    values = age_weighted_daly_values(
        table, _uniform_single_age_pop(), ["0"], year=2026, max_age=99
    )
    assert values.loc["0", "yll"] == pytest.approx(20.0)

    # With a population skewed young, the mean must move towards the younger band's value.
    skewed = _uniform_single_age_pop()
    skewed.loc[skewed["Age"] >= 50, "Pop"] = 100.0
    skewed_values = age_weighted_daly_values(
        table, skewed, ["0"], year=2026, max_age=99
    )
    assert skewed_values.loc["0", "yll"] > 20.0
    assert skewed_values.loc["0", "yll"] == pytest.approx(
        (50 * 1000 * 30.0 + 50 * 100 * 10.0) / (50 * 1000 + 50 * 100)
    )


def test_real_daly_table_remaps_onto_model_age_groups():
    table = inputs.load_age_daly_table(ISO3)
    values = age_weighted_daly_values(
        table, _uniform_single_age_pop(), ["0", "15", "35", "55", "70"], year=2026
    )
    assert list(values.index) == ["0", "15", "35", "55", "70"]
    # YLLs must still decrease with age after remapping.
    assert values["yll"].is_monotonic_decreasing


def test_missing_population_year_is_an_error():
    table = inputs.load_age_daly_table(ISO3)
    with pytest.raises(ValueError, match="No single-age population"):
        age_weighted_daly_values(table, _uniform_single_age_pop(), ["0"], year=1800)


def test_unsorted_age_groups_rejected():
    table = inputs.load_age_daly_table(ISO3)
    with pytest.raises(ValueError, match="ascending order"):
        age_weighted_daly_values(
            table, _uniform_single_age_pop(), ["15", "0"], year=2026
        )


# --------------------------------------------------------------------------------------
# DALY and cost streams against synthetic model outputs
# --------------------------------------------------------------------------------------

AGE_GROUPS = ["0", "15", "55"]
YEARS = list(range(2026, 2036))


def _synthetic_outputs(deaths=10.0, completions=100.0, symptomatic=50.0, on_treatment=20.0):
    data = {}
    for age in AGE_GROUPS:
        data[contract.TB_DEATHS_BY_AGE.format(age=age)] = deaths
        data[contract.TREATMENT_COMPLETIONS_BY_AGE.format(age=age)] = completions
        data[contract.SYMPTOMATIC_PERSON_TIME.format(age=age)] = symptomatic
        data[contract.TREATMENT_PERSON_TIME.format(age=age)] = on_treatment
    data[contract.TREATMENT_STARTS_ACTIVE] = 80.0
    data[contract.TREATMENT_STARTS_PASSIVE] = 40.0
    data["n_screened_cxr_xpert"] = 5000.0
    data["n_screen_positive_cxr_xpert"] = 60.0
    return pd.DataFrame(data, index=pd.Index(YEARS, name="year"))


def test_contract_validation_passes_for_complete_outputs():
    contract.validate_outputs(_synthetic_outputs(), AGE_GROUPS, ["cxr_xpert"])


def test_contract_validation_names_missing_outputs():
    outputs = _synthetic_outputs().drop(columns=[contract.TREATMENT_STARTS_ACTIVE])
    with pytest.raises(KeyError, match="tb_treatment_starts_active"):
        contract.validate_outputs(outputs, AGE_GROUPS, ["cxr_xpert"])


def test_daly_streams_sum_to_total():
    values = pd.DataFrame(
        {"yll": [30.0, 20.0, 10.0], "post_tb_daly": [5.0, 4.0, 2.0]}, index=AGE_GROUPS
    )
    streams = compute_daly_streams(_synthetic_outputs(), AGE_GROUPS, values, yld_weight=0.333)
    assert streams["total"].equals(streams[["ylls", "post_tb_dalys", "ylds"]].sum(axis=1))
    # 10 deaths per age group times the group's YLL value.
    assert streams["ylls"].iloc[0] == pytest.approx(10 * (30.0 + 20.0 + 10.0))
    # Person-time in both disabling states, weighted.
    assert streams["ylds"].iloc[0] == pytest.approx(0.333 * 3 * (50.0 + 20.0))


def test_total_dalys_discounting_reduces_later_years():
    values = pd.DataFrame(
        {"yll": [30.0, 20.0, 10.0], "post_tb_daly": [5.0, 4.0, 2.0]}, index=AGE_GROUPS
    )
    outputs = _synthetic_outputs()
    undiscounted = total_dalys(outputs, AGE_GROUPS, values, 0.333, discount_rate=0.0)
    discounted = total_dalys(outputs, AGE_GROUPS, values, 0.333, discount_rate=0.03)
    assert discounted["total"] < undiscounted["total"]
    assert undiscounted["total"] == pytest.approx(
        compute_daly_streams(outputs, AGE_GROUPS, values, 0.333)["total"].sum()
    )


def _cost_structure():
    return CostStructure(
        scenario="acf_cxr_xpert",
        components=[
            costs.CostComponent(
                "screening",
                inputs.UncertainQuantity("screening", 1.9668552086766822, 0.2 * 1.9668552086766822),
                "n_screened_cxr_xpert",
            ),
            costs.CostComponent(
                "diagnosis",
                inputs.UncertainQuantity("diagnosis", 3.0384615384615383, 0.2 * 3.0384615384615383),
                "n_screen_positive_cxr_xpert",
            ),
            costs.CostComponent(
                "patient",
                inputs.UncertainQuantity("patient", 578.910745066804, 0.2 * 578.910745066804),
                contract.TREATMENT_STARTS_ACTIVE,
                perspective="patient",
            ),
        ],
    )


def test_cost_streams_split_by_perspective():
    table = compute_cost_streams(_synthetic_outputs(), _cost_structure())
    assert table["total"].equals(table["total_provider"] + table["total_patient"])
    assert table["screening"].iloc[0] == pytest.approx(5000.0 * 1.9668552086766822)


def test_cost_structure_rejects_missing_quantity_output():
    structure = _cost_structure()
    outputs = _synthetic_outputs().drop(columns=["n_screened_cxr_xpert"])
    with pytest.raises(KeyError, match="n_screened_cxr_xpert"):
        compute_cost_streams(outputs, structure)


def test_probabilistic_draws_vary_and_centre_on_point_estimate():
    structure = _cost_structure()
    outputs = _synthetic_outputs()
    rng = np.random.default_rng(42)
    draws = sample_unit_costs(structure, 400, rng)
    totals = [
        total_costs(outputs, structure, unit_costs=d, discount_rate=0.03)["total"]
        for d in draws
    ]
    point = total_costs(outputs, structure, discount_rate=0.03)["total"]
    assert np.std(totals) > 0
    assert np.mean(totals) == pytest.approx(point, rel=0.05)


# --------------------------------------------------------------------------------------
# Cost-effectiveness
# --------------------------------------------------------------------------------------


def test_strong_dominance_identified():
    results = pd.DataFrame(
        {"cost": [100.0, 200.0, 150.0], "dalys": [1000.0, 1100.0, 800.0]},
        index=["cheap", "dominated", "good"],
    )
    frontier = efficiency_frontier(results)
    assert frontier.loc["dominated", "status"] == "dominated"
    assert frontier.loc["cheap", "status"] == "frontier"
    assert frontier.loc["good", "status"] == "frontier"


def test_icer_is_incremental_cost_per_daly_averted():
    results = pd.DataFrame(
        {"cost": [100.0, 300.0], "dalys": [1000.0, 900.0]}, index=["base", "intervention"]
    )
    frontier = efficiency_frontier(results)
    assert frontier.loc["intervention", "inc_dalys_averted"] == pytest.approx(100.0)
    assert frontier.loc["intervention", "icer"] == pytest.approx(2.0)


def test_extended_dominance_removed():
    # 'middle' has a worse ICER than the option above it, so it is extended-dominated.
    results = pd.DataFrame(
        {"cost": [0.0, 500.0, 600.0], "dalys": [1000.0, 910.0, 800.0]},
        index=["none", "middle", "best"],
    )
    frontier = efficiency_frontier(results)
    assert frontier.loc["middle", "status"] == "extended_dominated"
    assert frontier.loc["best", "status"] == "frontier"


def test_net_monetary_benefit_prefers_cheaper_and_healthier():
    better = net_monetary_benefit(100.0, 50.0, threshold=1000.0)
    worse = net_monetary_benefit(200.0, 60.0, threshold=1000.0)
    assert better > worse


def test_acceptability_curve_rows_sum_to_one():
    rng = np.random.default_rng(7)
    draws = {
        "none": pd.DataFrame({"cost": rng.normal(100, 5, 500), "dalys": rng.normal(1000, 20, 500)}),
        "screen": pd.DataFrame({"cost": rng.normal(400, 20, 500), "dalys": rng.normal(850, 20, 500)}),
    }
    curve = acceptability_curve(draws, thresholds=[0, 1000, 5000])
    assert np.allclose(curve.sum(axis=1), 1.0)
    # At zero willingness to pay, the cheaper option must dominate the curve.
    assert curve.loc[0, "none"] > curve.loc[0, "screen"]
    # At a high threshold, the health-improving option must win.
    assert curve.loc[5000, "screen"] > curve.loc[5000, "none"]


def test_acceptability_curve_rejects_mismatched_draw_counts():
    draws = {
        "a": pd.DataFrame({"cost": [1.0, 2.0], "dalys": [1.0, 2.0]}),
        "b": pd.DataFrame({"cost": [1.0], "dalys": [1.0]}),
    }
    with pytest.raises(ValueError, match="differing numbers of draws"):
        acceptability_curve(draws, thresholds=[1000])
