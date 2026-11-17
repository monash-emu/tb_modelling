"""Smoke test for the full pipeline, runnable now.

Vietnam cannot be calibrated yet — no targets, no Vietnam parameter file — so this exercises
the machinery end to end on Kiribati inputs instead: build, run, store with provenance,
reload, and cost. Run it after any change to the core.

    pixi run python notebooks/00_smoke_test.py
"""

import numpy as np
import pandas as pd

from tbcore.calibration import calibrate_and_run, iter_draws, normal_target
from tbcore.config import AnalysisConfig, ModelConfig
from tbcore.demography import get_population_over_time
from tbcore.economics import contract
from tbcore.economics.costs import CostComponent, CostStructure, total_costs
from tbcore.economics.dalys import age_weighted_daly_values, total_dalys
from tbcore.economics.inputs import UncertainQuantity, load_age_daly_table
from tbcore.economics.cea import acceptability_curve, efficiency_frontier
from tbcore.interventions import ScreeningProgram, ScreeningTools, Scenario
from tbcore.parameters import get_parameters_and_priors
from tbcore.runs import find_reusable_posterior, load_run, new_run_id, save_run

PROJECT = "smoke_test"
AGE_GROUPS = ("0", "5", "15", "35", "55", "70")

CONFIG = ModelConfig(
    iso3="KIR",
    start_time=1900,
    end_time=2040,
    scenario_start_time=2026,
    age_groups=AGE_GROUPS,
    pop_scaling=40483.0 / 119438.0,
    heterogeneous_mixing=False,
    output_set="economic",
)

ANALYSIS = AnalysisConfig(
    chains=2, cores=1, tune=50, draws=150, burn_in=50, full_runs_samples=20
)


def main() -> None:
    params, priors, tv_params = get_parameters_and_priors("kir")
    print(f"Loaded {len(params)} parameters, {len(priors)} priors")

    targets = [
        normal_target("notifications", pd.Series([250.0], index=[2020]), tol_perc=40.0),
        normal_target("tb_prevalence_per100k", pd.Series([800.0], index=[2024])),
    ]

    scenario = Scenario(
        sc_id="cxr_screening",
        sc_name="CXR screening at 70%",
        scr_prgs=[
            ScreeningProgram(
                name="cxr_all",
                scr_tool=ScreeningTools.CXR,
                start_time=2026,
                end_time=2027,
                coverage_prop=0.70,
                strata_coverage_multipliers={"age": {"0": 0.0}},
            )
        ],
    )

    reusable = find_reusable_posterior(PROJECT, CONFIG)
    idata = load_run(PROJECT, reusable.run_id).posterior() if reusable else None
    if reusable:
        print(f"Reusing posterior from {reusable.run_id} (config {reusable.config_hash})")

    print("Running calibration and scenarios...")
    results = calibrate_and_run(
        CONFIG, ANALYSIS, params, priors, targets, tv_params,
        scenarios=[scenario], idata=idata,
    )
    print(f"  timings: {results['timings']}")

    averted = results["differences"]["cxr_screening"]["TB_averted"]
    print(f"  TB episodes averted by 2040: {averted[0.5]:,.0f} "
          f"({averted[0.025]:,.0f}-{averted[0.975]:,.0f})")

    run_id = new_run_id("smoke")
    path = save_run(
        PROJECT, run_id, CONFIG, ANALYSIS,
        idata=results["idata"], quantiles=results["quantiles"],
        description="pipeline smoke test on Kiribati inputs",
        timings=results["timings"],
    )
    print(f"Saved run to {path}")

    stored = load_run(PROJECT, run_id)
    print(f"  commit {stored.manifest.git_commit[:8]}, dirty={stored.manifest.git_dirty}")

    cost_and_daly_demo(results)


def cost_and_daly_demo(results: dict) -> None:
    """Cost the scenarios from stored outputs, using illustrative Vietnam unit costs.

    The unit costs are Vietnamese and the epidemiology is Kiribati's, so the numbers mean
    nothing. The point is that costing runs entirely off derived outputs, with no model
    rebuild — which is what makes changing a cost assumption during peer review cheap.

    Costs and DALYs are computed per posterior draw, paired across scenarios by draw, so
    that incremental results carry a credible interval rather than being a ratio of medians.
    """
    print("\nCosting from stored outputs (illustrative only):")
    daly_table = load_age_daly_table("vnm")
    single_age_pop, _ = get_population_over_time("KIR", list(AGE_GROUPS))
    daly_values = age_weighted_daly_values(daly_table, single_age_pop, list(AGE_GROUPS), 2026)

    per_draw, summary = {}, {}
    for name, run in results["full_runs"].items():
        structure = _cost_structure(name)
        rows = []
        for _, outputs in iter_draws(run.results):
            contract.validate_outputs(
                outputs, list(AGE_GROUPS), ["cxr"] if name != "baseline" else []
            )
            costs = total_costs(
                outputs, structure, discount_rate=0.03, reference_year=2026,
                start_year=2026, end_year=2040,
            )
            dalys = total_dalys(
                outputs, list(AGE_GROUPS), daly_values, yld_weight=0.333,
                discount_rate=0.03, reference_year=2026, start_year=2026, end_year=2040,
            )
            rows.append({"cost": costs["total"], "dalys": dalys["total"]})

        per_draw[name] = pd.DataFrame(rows)
        summary[name] = per_draw[name].median()
        print(
            f"  {name:16s} cost ${summary[name]['cost']:>14,.0f}"
            f"   DALYs {summary[name]['dalys']:>12,.0f}"
            f"   ({len(rows)} draws)"
        )

    frontier = efficiency_frontier(pd.DataFrame(summary).T)
    print("\nIncremental analysis (at median):")
    print(frontier[["cost", "dalys", "status", "icer"]].to_string())

    if "cxr_screening" in per_draw:
        incremental_cost = per_draw["cxr_screening"]["cost"] - per_draw["baseline"]["cost"]
        averted = per_draw["baseline"]["dalys"] - per_draw["cxr_screening"]["dalys"]
        icer = incremental_cost / averted
        print(
            f"\nICER across draws: {icer.median():,.0f} "
            f"({icer.quantile(0.025):,.0f}-{icer.quantile(0.975):,.0f}) USD per DALY averted"
        )

    curve = acceptability_curve(per_draw, thresholds=[0, 500, 1000, 2500, 5000])
    print("\nProbability each option is optimal, by willingness to pay:")
    print(curve.to_string(float_format=lambda v: f"{v:.2f}"))


def _cost_structure(scenario: str) -> CostStructure:
    components = [
        CostComponent(
            "treatment", UncertainQuantity("treatment", 123.84, 24.77),
            contract.TREATMENT_STARTS_PASSIVE,
        ),
        CostComponent(
            "patient", UncertainQuantity("patient", 1359.01, 271.80),
            contract.TREATMENT_STARTS_PASSIVE, perspective="patient",
        ),
    ]
    if scenario != "baseline":
        components.append(
            CostComponent(
                "screening", UncertainQuantity("screening", 1.97, 0.39), "n_screened_cxr"
            )
        )
    return CostStructure(scenario=scenario, components=components)


if __name__ == "__main__":
    main()
