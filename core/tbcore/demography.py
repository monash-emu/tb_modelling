"""Demographic inputs from UN World Population Prospects.

Population and all-cause mortality are read from multi-country extracts in ``data/shared``.
Both currently cover IDN, KIR, PHL and VNM.

Fertility is *not* handled here. It is needed only for the parent-child component of the
age-mixing matrix, is available for KIR alone, and mixing is homogeneous for now — see
:mod:`tbcore.mixing`.
"""

import pandas as pd
from summer2.functions import time as stf

from tbcore.paths import SHARED_DATA

POPULATION_FILE = "un_population.csv"
MORTALITY_FILE = "un_mortality.csv"

#: UN data are reported in thousands.
THOUSANDS = 1000.0

MAX_AGE = 120


def available_countries(filename: str = POPULATION_FILE) -> list[str]:
    """ISO3 codes present in the shared demographic extract."""
    codes = pd.read_csv(SHARED_DATA / filename, usecols=["ISO3_code"])["ISO3_code"].unique()
    return sorted(str(c) for c in codes)


def _check_country(iso3: str, data: pd.DataFrame, source: str) -> None:
    if data.empty:
        raise ValueError(
            f"No {source} data for '{iso3}'. Available: {available_countries()}. "
            "Adding a country means extending the shared UN extracts."
        )


def get_population_over_time(
    iso3: str, age_groups: list[str], max_age: int = MAX_AGE, scaling_factor: float = 1.0
):
    """Population size over time, by single year of age and by model age group.

    Parameters
    ----------
    iso3
        Country code.
    age_groups
        Model age group lower bounds, ascending.
    scaling_factor
        Multiplier for subnational models. 1.0 for national runs.

    Returns
    -------
    single_age_pop : pd.DataFrame
        Columns ``Time``, ``Age``, ``Pop``. Also consumed by the economics layer, which uses
        the within-group age distribution to weight age-specific DALY values.
    grouped_pop : pd.DataFrame
        Time-indexed, one column per model age group.
    """
    raw = pd.read_csv(SHARED_DATA / POPULATION_FILE)
    raw = raw[raw["ISO3_code"] == iso3][["Time", "AgeGrp", "PopTotal"]]
    _check_country(iso3, raw, "population")

    # Spread each reported band evenly across the single years of age it covers.
    rows = []
    for _, record in raw.iterrows():
        pop = record["PopTotal"] * THOUSANDS * scaling_factor
        low, high = _parse_reported_band(record["AgeGrp"], max_age)
        n_ages = high - low + 1
        rows.extend({"Time": record["Time"], "Age": age, "Pop": pop / n_ages} for age in range(low, high + 1))

    single_age_pop = pd.DataFrame(rows)
    grouped_pop = _aggregate_to_groups(single_age_pop, age_groups, max_age)
    return single_age_pop, grouped_pop


def _parse_reported_band(label, max_age: int) -> tuple:
    if isinstance(label, str) and label.endswith("+"):
        return int(label[:-1]), max_age
    low, high = map(int, str(label).split("-"))
    return low, high


def _aggregate_to_groups(single_age_pop: pd.DataFrame, age_groups: list[str], max_age: int):
    bounds = group_bounds(age_groups, max_age)
    out = {}
    for year, year_data in single_age_pop.groupby("Time"):
        by_age = year_data.set_index("Age")["Pop"]
        out[year] = {
            label: by_age.loc[(by_age.index >= low) & (by_age.index <= high)].sum()
            for low, high, label in bounds
        }
    return pd.DataFrame(out).T.sort_index().rename_axis("Time")


def group_bounds(age_groups: list[str], max_age: int = MAX_AGE) -> list[tuple]:
    """Inclusive lower and upper bound of each model age group."""
    lowers = [int(a) for a in age_groups]
    if lowers != sorted(lowers):
        raise ValueError(f"Age groups must be ascending, got {age_groups}")
    return [
        (low, lowers[i + 1] - 1 if i < len(lowers) - 1 else max_age, age_groups[i])
        for i, low in enumerate(lowers)
    ]


def get_death_rates_by_age(
    iso3: str, age_groups: list[str], start_time: int, grouped_pop: pd.DataFrame
) -> dict:
    """Age-specific all-cause death rate functions of time.

    Deaths reported by the UN are divided by the modelled population in the same age group,
    giving a per-capita rate. Note that the rate is therefore insensitive to
    ``pop_scaling``: scaling cancels between numerator and denominator only if the
    population used here is scaled the same way, which it is, since ``grouped_pop`` is passed
    in rather than re-read.

    Returns
    -------
    dict
        Age group label to a sigmoidal interpolation function of time.
    """
    bins = [int(a) for a in age_groups]
    last_bin = bins[-1]

    raw = pd.read_csv(SHARED_DATA / MORTALITY_FILE)
    raw = raw[(raw["ISO3_code"] == iso3) & (raw["Time"] >= start_time)]
    _check_country(iso3, raw, "mortality")
    raw = raw[["Time", "AgeGrp", "DeathTotal"]].copy()
    raw["DeathTotal"] *= THOUSANDS

    def assign_bin(age_label):
        age = last_bin if str(age_label) == "100+" else int(age_label)
        if age >= last_bin:
            return last_bin
        for i in range(len(bins) - 1):
            if bins[i] <= age < bins[i + 1]:
                return bins[i]
        return None

    raw["age_group"] = raw["AgeGrp"].apply(assign_bin)
    deaths = (
        raw.groupby(["Time", "age_group"], as_index=False)
        .agg({"DeathTotal": "sum"})
        .pivot(index="Time", columns="age_group", values="DeathTotal")
    )
    deaths.columns = deaths.columns.astype(str)

    death_rates = deaths.div(grouped_pop, axis=0).dropna()
    if death_rates.empty:
        raise ValueError(
            f"No overlapping years between mortality data and population data for {iso3}."
        )

    return {
        age: stf.get_sigmoidal_interpolation_function(
            death_rates.index, death_rates[age].values
        )
        for age in age_groups
    }
