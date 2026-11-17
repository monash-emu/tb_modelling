"""The contract between the model and the economic layer.

Economic analysis is post-processing: it consumes saved derived outputs and never touches
the model. That only works if both sides agree on exactly which derived outputs must exist
and what they mean. This module is that agreement, stated once.

``tbcore.outputs.resource_use`` requests these outputs; the functions in
``tbcore.economics`` consume them. A mismatch surfaces here as a clear error rather than as
a silent zero in a cost total.
"""

#: Persons screened per year by a given screening tool. Drives "screening, per person
#: screened" provider costs.
N_SCREENED = "n_screened_{tool}"

#: Screen-positives per year by tool. Drives "diagnosis, per positive case" provider costs,
#: which differ between Xpert-based and Pluslife-based algorithms.
N_SCREEN_POSITIVE = "n_screen_positive_{tool}"

#: Diagnostic tests performed per year by test type. Not used for costing directly where
#: unit costs are expressed per person screened, but required for resource-use reporting and
#: for algorithms costed test by test.
N_TESTS = "n_tests_{test}"

#: Treatment starts per year, split by how the case was found. The split is essential rather
#: than cosmetic: patient costs differ roughly two-fold between actively detected and
#: passively detected cases, and that difference is a principal driver of cost-effectiveness.
TREATMENT_STARTS_ACTIVE = "tb_treatment_starts_active"
TREATMENT_STARTS_PASSIVE = "tb_treatment_starts_passive"

#: Person-time in symptomatic TB, by age. Drives symptomatic-TB YLDs.
SYMPTOMATIC_PERSON_TIME = "prev_symptomaticXage_{age}"

#: Person-time on TB treatment, by age. Drives on-treatment YLDs.
TREATMENT_PERSON_TIME = "n_on_treatmentXage_{age}"

#: TB deaths per year by age. Drives YLLs.
TB_DEATHS_BY_AGE = "tb_mortalityXage_{age}"

#: Successful treatment completions per year by age. Drives post-TB DALYs, which accrue to
#: survivors of TB treatment.
TREATMENT_COMPLETIONS_BY_AGE = "tx_completionsXage_{age}"


def required_outputs(age_groups: list[str], tools: list[str]) -> list[str]:
    """Every derived output the economic layer needs, for a given model configuration."""
    names = [TREATMENT_STARTS_ACTIVE, TREATMENT_STARTS_PASSIVE]
    for age in age_groups:
        names += [
            SYMPTOMATIC_PERSON_TIME.format(age=age),
            TREATMENT_PERSON_TIME.format(age=age),
            TB_DEATHS_BY_AGE.format(age=age),
            TREATMENT_COMPLETIONS_BY_AGE.format(age=age),
        ]
    for tool in tools:
        names += [N_SCREENED.format(tool=tool), N_SCREEN_POSITIVE.format(tool=tool)]
    return names


def validate_outputs(outputs_df, age_groups: list[str], tools: list[str]) -> None:
    """Raise if the derived-output dataframe is missing anything the cost engine needs."""
    missing = [n for n in required_outputs(age_groups, tools) if n not in outputs_df.columns]
    if missing:
        raise KeyError(
            f"Derived outputs are missing {len(missing)} fields required for economic "
            f"analysis: {missing[:8]}{' ...' if len(missing) > 8 else ''}. "
            "The model was probably built with an output set that excludes resource use; "
            "rebuild with output_set='economic'."
        )
