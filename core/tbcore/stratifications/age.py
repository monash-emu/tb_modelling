"""Age stratification.

Always applied, and applied before any other stratification, because it carries the mixing
matrix and because the ageing flows it creates must exist before other dimensions subdivide
the compartments further.

Age is not a :class:`~tbcore.stratifications.base.StratificationModule`: it uses
``summer2.AgeStratification`` rather than a plain ``Stratification``, and it needs the
screening programmes and treatment-outcome functions, so it keeps its own signature.
"""

from summer2 import AgeStratification, CompartmentalModel
from summer2.parameters import Parameter

from tbcore.compartments import ACTIVE_COMPS, COMPARTMENTS, is_asymptomatic, is_low_infectious

#: Age breaks required for compatibility with age-specific early-infection parameters
#: (``progression_rate_age{0,5,15}``, ``containment_rate_age{0,5,15}``), with the BCG effect
#: in under-15s, and with the contact matrix construction.
REQUIRED_AGE_BREAKS = ["0", "5", "15"]

#: Age below which individuals are treated as non-infectious and as receiving BCG protection.
CHILD_AGE_LIMIT = 15


def latency_age_category(age: str) -> str:
    """Map a model age break to the age category of the early-infection parameters."""
    age_int = int(age)
    if age_int < 5:
        return "age0"
    if age_int < CHILD_AGE_LIMIT:
        return "age5"
    return "age15"


def validate_age_groups(age_groups: list[str]) -> None:
    missing = [break_ for break_ in REQUIRED_AGE_BREAKS if break_ not in age_groups]
    if missing:
        raise ValueError(
            f"Age breaks {missing} are required but missing from {age_groups}. "
            "They are needed because early-infection parameters are specified for 0-4, 5-14 "
            "and 15+, and because BCG protection and infectiousness switch at age 15."
        )


def stratify_by_age(
    model: CompartmentalModel,
    age_groups: list[str],
    neg_tx_outcome_funcs: dict,
    screening_programs: list,
    age_mixing_matrix=None,
) -> None:
    """Apply age stratification to the model.

    Parameters
    ----------
    model
        The unstratified model, with natural history and detection flows already added.
    age_groups
        Lower bounds of each age stratum, as strings, in ascending order.
    neg_tx_outcome_funcs
        Age-keyed dict of ``{"relapse": ..., "death": ...}`` functions. Treatment outcomes are
        age-specific because background mortality competes with treatment outcomes.
    screening_programs
        Active screening programmes, whose age-eligibility multipliers are applied here.
    age_mixing_matrix
        Mixing matrix function, or ``None`` for homogeneous mixing.
    """
    validate_age_groups(age_groups)

    age_strat = AgeStratification(name="age", strata=age_groups, compartments=COMPARTMENTS)

    if age_mixing_matrix is not None:
        age_strat.set_mixing_matrix(age_mixing_matrix)

    # Progression out of the incipient state, and containment, are strongly age-dependent:
    # young children progress fast and contain poorly.
    for flow_name in ["progression_lowinf", "progression_inf", "containment"]:
        param_stem = "containment" if flow_name == "containment" else "progression"
        age_strat.set_flow_adjustments(
            flow_name=flow_name,
            adjustments={
                age: Parameter(f"{param_stem}_rate_{latency_age_category(age)}")
                for age in age_groups
            },
        )

    # Reduced susceptibility in children, capturing BCG protection.
    age_strat.set_flow_adjustments(
        flow_name="infection_from_mtb_naive",
        adjustments={
            age: (Parameter("rel_sus_children") if int(age) < CHILD_AGE_LIMIT else 1.0)
            for age in age_groups
        },
    )

    # Relapse competes with age-specific background mortality during treatment.
    age_strat.set_flow_adjustments(
        "tx_relapse", {age: funcs["relapse"] for age, funcs in neg_tx_outcome_funcs.items()}
    )

    # Asymptomatic and low-infectiousness disease transmit less; children are treated as
    # non-infectious.
    for compartment in ACTIVE_COMPS:
        adjuster = 1.0
        if is_low_infectious(compartment):
            adjuster *= Parameter("rel_infectiousness_lowinf")
        if is_asymptomatic(compartment):
            adjuster *= Parameter("rel_infectiousness_subclin")
        age_strat.add_infectiousness_adjustments(
            compartment,
            {age: (0.0 if int(age) < CHILD_AGE_LIMIT else adjuster) for age in age_groups},
        )

    # Age eligibility for each screening programme.
    for program in screening_programs:
        age_multipliers = program.strata_coverage_multipliers.get("age", {})
        unknown = set(age_multipliers) - set(age_groups)
        if unknown:
            raise ValueError(
                f"Screening programme '{program.name}' sets age multipliers for {sorted(unknown)}, "
                f"which are not model age breaks {age_groups}."
            )
        for source_comp in program.scr_tool["sensitivities"]:
            age_strat.set_flow_adjustments(
                f"{program.name}_{source_comp}",
                {age: age_multipliers.get(age, 1.0) for age in age_groups},
            )

    model.stratify_with(age_strat)
