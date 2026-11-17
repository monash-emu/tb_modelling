"""Bayesian calibration and scenario execution.

Calibration targets are deliberately *not* defined here: they are project data, and the
published model's habit of hard-coding them in the shared runner is what made that code
impossible to reuse across countries. Projects supply targets; this module supplies the
machinery to fit and to run scenarios against a posterior.
"""

from time import time

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
from estival import targets as est
from estival.model import BayesianCompartmentalModel
from estival.sampling import tools as esamp
from estival.wrappers import pymc as epm

from tbcore.model import get_tb_model

SAMPLERS = {"DEMetropolisZ": "DEMetropolisZ", "DEMetropolis": "DEMetropolis"}

DEFAULT_QUANTILES = [0.025, 0.25, 0.5, 0.75, 0.975]


def normal_target(name: str, data: pd.Series, tol_perc: float = 20.0) -> est.NormalTarget:
    """Normal likelihood whose width is a tolerance around the data mean.

    A tolerance of 20% places 95% of the density within ±20% of the mean, which is a more
    interpretable way to state how strongly a target should bind than a raw standard
    deviation.
    """
    if data.empty:
        raise ValueError(f"Target '{name}' has no data")
    return est.NormalTarget(
        name=name, data=data, stdev=(tol_perc / 100.0) / 1.96 * data.mean()
    )


def build_bcm(
    config,
    params: dict,
    priors: list,
    targets: list,
    tv_params: dict,
    screening_programs: list = None,
    lever=None,
) -> BayesianCompartmentalModel:
    """Assemble a model, its parameters, priors and targets into a calibration object."""
    model = get_tb_model(config, tv_params, screening_programs, lever)
    _check_targets_against_outputs(model, targets)
    return BayesianCompartmentalModel(model, params, priors, targets)


def _check_targets_against_outputs(model, targets: list) -> None:
    """Fail early when a target names an output the model does not produce.

    Without this the mismatch surfaces deep inside the sampler, after the expensive part.
    """
    requested = set(model._derived_output_requests)  # noqa: SLF001
    missing = [t.name for t in targets if t.name not in requested]
    if missing:
        raise KeyError(
            f"Calibration targets reference outputs the model does not produce: {missing}. "
            "Check the output set, the age aggregates and the stratifications in the model "
            "configuration."
        )


def run_metropolis(bcm, analysis_config) -> az.InferenceData:
    """Metropolis sampling via pymc."""
    if analysis_config.sampler not in SAMPLERS:
        raise ValueError(
            f"Sampler '{analysis_config.sampler}' not supported; choose from {list(SAMPLERS)}"
        )
    sampler = getattr(pm, analysis_config.sampler)

    with pm.Model():
        variables = epm.use_model(bcm)
        idata = pm.sample(
            step=[sampler(variables)],
            draws=analysis_config.draws,
            tune=analysis_config.tune,
            cores=analysis_config.cores,
            chains=analysis_config.chains,
            progressbar=False,
        )
    return idata


def sample_from_posterior(
    idata: az.InferenceData, burn_in: int, n_samples: int, random_seed: int = None
):
    """Draw parameter sets from the posterior after discarding burn-in.

    Subsampling is random, so a seed is required for reproducibility: without one, re-running
    an analysis against a stored posterior gives slightly different numbers, which is exactly
    the situation a reviewer re-running the code must not encounter.
    """
    chain_length = idata.sample_stats.sizes["draw"]
    if burn_in >= chain_length:
        raise ValueError(f"Burn-in {burn_in} exceeds chain length {chain_length}")
    available = chain_length - burn_in
    if n_samples > available:
        raise ValueError(
            f"Requested {n_samples} samples but only {available} draws remain after burn-in"
        )
    burnt = idata.sel(draw=range(burn_in, chain_length))
    rng = np.random.default_rng(random_seed) if random_seed is not None else None
    return az.extract(burnt, num_samples=n_samples, rng=rng)


def run_scenarios(
    bcm_dict: dict,
    idata: az.InferenceData,
    burn_in: int,
    n_samples: int,
    quantiles: list = None,
    random_seed: int = None,
) -> tuple:
    """Run every scenario across posterior samples.

    All scenarios use the *same* posterior draws, so differences between them reflect the
    intervention rather than sampling noise, and paired comparisons such as cases averted
    are valid draw by draw.

    Returns
    -------
    full_runs : dict
        Scenario name to the full set of per-draw results.
    quantile_dfs : dict
        Scenario name to quantiles across draws.
    """
    quantiles = quantiles or DEFAULT_QUANTILES
    parameter_sets = sample_from_posterior(idata, burn_in, n_samples, random_seed)

    full_runs, quantile_dfs = {}, {}
    for name, bcm in bcm_dict.items():
        results = esamp.model_results_for_samples(parameter_sets, bcm, include_extras=False)
        quantile_df = esamp.quantiles_for_results(results.results, quantiles)
        # Float column labels are not parquet-compatible.
        quantile_df.columns = quantile_df.columns.set_levels(
            [str(q) for q in quantile_df.columns.levels[1]], level=1
        )
        full_runs[name] = results
        quantile_dfs[name] = quantile_df

    return full_runs, quantile_dfs


def draw_keys(results: pd.DataFrame) -> list:
    """``(chain, draw)`` identifiers present in a scenario's full results."""
    return list(dict.fromkeys(zip(*[results.columns.get_level_values(level) for level in ("chain", "draw")])))


def outputs_for_draw(results: pd.DataFrame, key: tuple) -> pd.DataFrame:
    """Time-indexed derived outputs for a single posterior draw.

    Full scenario results are time-indexed with ``(variable, chain, draw)`` columns. The
    economic layer consumes one scenario-draw at a time, so that cost and DALY totals can be
    produced per draw and summarised as credible intervals.
    """
    chain, draw = key
    return results.xs((chain, draw), level=("chain", "draw"), axis=1)


def iter_draws(results: pd.DataFrame):
    """Yield ``(key, outputs_df)`` for every posterior draw in a scenario's results."""
    for key in draw_keys(results):
        yield key, outputs_for_draw(results, key)


def difference_quantiles(
    full_runs: dict,
    outputs: dict = None,
    reference: str = "baseline",
    at_year: int = None,
    quantiles: list = None,
) -> dict:
    """Paired differences between each scenario and a reference, by posterior draw.

    Differencing within a draw rather than between summary statistics is what makes the
    credible interval on "cases averted" meaningful.

    Parameters
    ----------
    outputs
        Result column name to output label, e.g.
        ``{"cum_tb_incidence": "TB_averted"}``. Defaults to cumulative incidence and
        mortality.
    at_year
        Year at which to evaluate. Defaults to the last simulated year.
    """
    outputs = outputs or {"cum_tb_incidence": "TB_averted", "cum_tb_mortality": "deaths_averted"}
    quantiles = quantiles or DEFAULT_QUANTILES

    if reference not in full_runs:
        raise KeyError(f"Reference scenario '{reference}' not among {list(full_runs)}")

    reference_results = full_runs[reference].results
    year = at_year if at_year is not None else reference_results.index.get_level_values(0).max()
    reference_at_year = reference_results.loc[year]

    differences = {}
    for name, run in full_runs.items():
        if name == reference:
            continue
        scenario_at_year = run.results.loc[year]
        absolute = reference_at_year - scenario_at_year
        relative = absolute / reference_at_year

        table = pd.DataFrame(
            index=quantiles,
            data={
                label: absolute[column].quantile(quantiles)
                for column, label in outputs.items()
            },
        )
        relative_table = pd.DataFrame(
            index=quantiles,
            data={
                f"{label}_relative": relative[column].quantile(quantiles)
                for column, label in outputs.items()
            },
        )
        differences[name] = pd.concat([table, relative_table], axis=1)

    return differences


def calibrate_and_run(
    config,
    analysis_config,
    params: dict,
    priors: list,
    targets: list,
    tv_params: dict,
    scenarios: list = None,
    idata: az.InferenceData = None,
) -> dict:
    """Calibrate if needed, then run every scenario against the posterior.

    Passing an existing ``idata`` skips calibration entirely — the intended path when only
    scenarios or cost assumptions have changed. See
    :func:`tbcore.runs.find_reusable_posterior`.

    Returns
    -------
    dict
        ``idata``, ``full_runs``, ``quantiles``, ``differences`` and ``timings``.
    """
    scenarios = scenarios or []
    scenario_ids = [s.sc_id for s in scenarios]
    if len(scenario_ids) != len(set(scenario_ids)):
        raise ValueError(f"Scenario ids must be unique, got {scenario_ids}")
    if "baseline" in scenario_ids:
        raise ValueError("'baseline' is reserved for the no-intervention comparator")

    timings = {}
    baseline_bcm = build_bcm(config, params, priors, targets, tv_params)

    if idata is None:
        start = time()
        idata = run_metropolis(baseline_bcm, analysis_config)
        timings["calibration_seconds"] = round(time() - start)

    start = time()
    bcm_dict = {"baseline": baseline_bcm}
    for scenario in scenarios:
        bcm_dict[scenario.sc_id] = build_bcm(
            config,
            params | scenario.params_ow,
            priors,
            targets,
            tv_params,
            screening_programs=scenario.scr_prgs,
            lever=scenario.lever,
        )

    full_runs, quantiles = run_scenarios(
        bcm_dict,
        idata,
        analysis_config.burn_in,
        analysis_config.full_runs_samples,
        random_seed=analysis_config.random_seed,
    )
    timings["scenario_seconds"] = round(time() - start)

    return {
        "idata": idata,
        "full_runs": full_runs,
        "quantiles": quantiles,
        "differences": difference_quantiles(full_runs, at_year=config.end_time),
        "timings": timings,
    }
