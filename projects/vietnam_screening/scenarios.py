"""Screening scenarios for the Vietnam cost-effectiveness analysis.

The four active case-finding algorithms are those named in the economic input spreadsheet:
``ACF_Xpert``, ``ACF_CXR_Xpert``, ``ACF_Pluslife`` and ``ACF_CXR_Pluslife``, compared against
no screening.

The two Pluslife arms depend on tongue-swab sensitivity parameters that are placeholders in
the inherited parameter file. They are defined here so the structure is reviewable, but they
must not be interpreted until those values are replaced.
"""

from tbcore.compartments import ACTIVE_COMPS
from tbcore.interventions import Scenario, ScreeningProgram, ScreeningTool, ScreeningTools
from summer2.parameters import Parameter

from projects.vietnam_screening.config import SCREENING_END, SCREENING_START

#: Proportion of the eligible population screened over the campaign. Provisional — the
#: coverage actually achievable at national scale has not been specified.
DEFAULT_COVERAGE = 0.70

#: Screening is not offered to young children in any arm.
EXCLUDED_AGES = {"0": 0.0, "5": 0.0}


def _xpert_only_tool() -> ScreeningTool:
    """Universal Xpert without chest X-ray triage: every participant is tested."""
    return ScreeningTool(
        name="acf_xpert",
        sensitivities={s: Parameter(f"prev_se_{s}_pearl") for s in ACTIVE_COMPS},
        dest_comp="treatment",
        tests_per_person={"symptom_screen": 1.0, "xpert": 1.0},
    )


def _cxr_xpert_tool() -> ScreeningTool:
    """Chest X-ray triage with Xpert confirmation of radiographic positives.

    Only X-ray positives proceed to Xpert, so the number of Xpert tests per person screened
    is the X-ray positivity proportion rather than one.
    """
    return ScreeningTool(
        name="acf_cxr_xpert",
        sensitivities={
            s: Parameter(f"prev_se_{s}_cxr") * Parameter(f"prev_se_{s}_pearl")
            for s in ACTIVE_COMPS
        },
        dest_comp="treatment",
        tests_per_person={
            "symptom_screen": 1.0,
            "cxr": 1.0,
            # FIXME: needs a cxr_positivity_prop parameter reflecting X-ray positivity in
            # the screened population, not just among those with TB.
            "xpert": Parameter("cxr_positivity_prop"),
        },
    )


def _pluslife_tool() -> ScreeningTool:
    # FIXME: prev_se_*_plts values are placeholders flagged `skip` in the inherited
    # parameter file. Do not interpret Pluslife arms until they are replaced.
    return ScreeningTool(
        name="acf_pluslife",
        sensitivities={s: Parameter(f"prev_se_{s}_plts") for s in ACTIVE_COMPS},
        dest_comp="treatment",
        tests_per_person={"symptom_screen": 1.0, "plts": 1.0},
    )


def _cxr_pluslife_tool() -> ScreeningTool:
    # FIXME: as above, plus the same X-ray positivity question as the CXR/Xpert arm.
    return ScreeningTool(
        name="acf_cxr_pluslife",
        sensitivities={
            s: Parameter(f"prev_se_{s}_cxr") * Parameter(f"prev_se_{s}_plts")
            for s in ACTIVE_COMPS
        },
        dest_comp="treatment",
        tests_per_person={
            "symptom_screen": 1.0,
            "cxr": 1.0,
            "plts": Parameter("cxr_positivity_prop"),
        },
    )


ALGORITHMS = {
    "acf_xpert": _xpert_only_tool,
    "acf_cxr_xpert": _cxr_xpert_tool,
    "acf_pluslife": _pluslife_tool,
    "acf_cxr_pluslife": _cxr_pluslife_tool,
}

#: Arms whose parameters are still placeholders.
PROVISIONAL_ALGORITHMS = {"acf_pluslife", "acf_cxr_pluslife"}


def build_scenario(algorithm: str, coverage: float = DEFAULT_COVERAGE) -> Scenario:
    """One screening arm at a given coverage."""
    if algorithm not in ALGORITHMS:
        raise ValueError(f"Unknown algorithm '{algorithm}'; choose from {list(ALGORITHMS)}")

    tool = ALGORITHMS[algorithm]()
    program = ScreeningProgram(
        name=algorithm,
        scr_tool=tool,
        start_time=SCREENING_START,
        end_time=SCREENING_END,
        coverage_prop=coverage,
        strata_coverage_multipliers={"age": dict(EXCLUDED_AGES)},
    )
    return Scenario(
        sc_id=f"{algorithm}_cov{int(100 * coverage)}",
        sc_name=f"{algorithm.replace('_', ' ').upper()} at {int(100 * coverage)}%",
        scr_prgs=[program],
        description=f"Mass screening using the {algorithm} algorithm.",
    )


def all_scenarios(coverage: float = DEFAULT_COVERAGE, include_provisional: bool = False) -> list:
    """Every screening arm. The no-screening comparator is the baseline and is not listed.

    Provisional arms are excluded by default so that an accidental full run cannot produce
    interpretable-looking Pluslife results from placeholder sensitivities.
    """
    algorithms = [
        name
        for name in ALGORITHMS
        if include_provisional or name not in PROVISIONAL_ALGORITHMS
    ]
    return [build_scenario(name, coverage) for name in algorithms]
