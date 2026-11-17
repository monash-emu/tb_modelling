"""Parameter and prior loading.

Parameters live in a spreadsheet per country, not in code, so that the appendix parameter
tables can be generated from the same source that the model reads. The spreadsheet is the
single source of truth for values, priors, sources, LaTeX symbols and units.

Sheet ``constant`` holds non-time-variant parameters: a value for every parameter, plus
distribution and bounds for those estimated during calibration. Sheet ``time_variant`` holds
year-indexed series such as treatment success.
"""

import pandas as pd
from estival import priors as esp

from tbcore.paths import country_data

PARAMS_FILE = "parameters.xlsx"


def get_parameters_and_priors(iso3: str, filename: str = PARAMS_FILE):
    """Read fixed parameters, prior distributions and time-variant series.

    Returns
    -------
    params : dict
        Parameter name to value, for every parameter including those with priors. Values for
        calibrated parameters act as defaults for single runs.
    priors : list
        ``estival`` prior objects for parameters with a distribution specified.
    tv_params : dict
        Parameter name to a year-indexed series.
    """
    path = country_data(iso3) / filename
    if not path.exists():
        raise FileNotFoundError(
            f"No parameter file for {iso3} at {path}. Each country needs its own, since "
            "programmatic parameters such as treatment success are not transferable."
        )

    constant = pd.read_excel(path, sheet_name="constant")
    constant = constant.where(pd.notna(constant), None)

    missing_values = constant[constant["value"].isna()]["parameter"].tolist()
    if missing_values:
        raise ValueError(f"Parameters with no value in {path}: {missing_values}")

    params = dict(zip(constant["parameter"], constant["value"]))
    priors = [
        build_prior(row["parameter"], row["distribution"], row["distri_param1"], row["distri_param2"])
        for _, row in constant[constant["distribution"].notnull()].iterrows()
    ]

    time_variant = pd.read_excel(path, sheet_name="time_variant", index_col=0)
    tv_params = {col: time_variant[col].dropna() for col in time_variant.columns}

    return params, priors, tv_params


def build_prior(name: str, distribution: str, param_1, param_2=None):
    """Build an ``estival`` prior from a spreadsheet row."""
    if distribution == "uniform":
        if param_2 is None:
            raise ValueError(f"Uniform prior '{name}' needs two bounds")
        if param_2 <= param_1:
            raise ValueError(f"Uniform prior '{name}' has upper bound <= lower bound")
        return esp.UniformPrior(name, [param_1, param_2])
    if distribution == "beta":
        return esp.BetaPrior(name, param_1, param_2)
    if distribution == "gamma":
        return esp.GammaPrior(name, param_1, param_2)
    raise ValueError(
        f"Prior distribution '{distribution}' for '{name}' is not supported. "
        "Supported: uniform, beta, gamma."
    )


def check_required_parameters(params: dict, required: list) -> None:
    """Raise if any parameter the model needs is absent from the spreadsheet."""
    missing = [name for name in required if name not in params]
    if missing:
        raise KeyError(f"Parameter spreadsheet is missing: {missing}")
