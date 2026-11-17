"""Cost structures for the Vietnam cost-effectiveness analysis.

Maps rows of ``data/vnm/economic_inputs.xlsx`` onto the model outputs they are applied to.
The mapping is written out explicitly rather than parsed from the spreadsheet's prose
``applies_to`` column, so that what is being costed is visible and reviewable.

INTERPRETATION REQUIRING CONFIRMATION
-------------------------------------
The spreadsheet labels unit costs by *scenario*, but the model produces quantities split by
*detection route*. Two rows have to be reinterpreted to bridge that, and both choices
materially change the result:

1. **Patient costs.** The sheet gives 578.91 for "ACF scenarios" and 1359.01 for the
   "No screening scenario". These are read here as costs per *actively* and *passively*
   detected case respectively, applied in every scenario according to how each case was
   found. The alternative reading — every case in a screening scenario incurring the lower
   cost, including those still found passively — would overstate the savings from screening.

2. **Routine diagnosis cost.** The sheet gives 324.95 for "diagnosis, per diagnosed case"
   under the "No screening scenario". It is applied here to passively detected cases in
   *every* scenario, since routine diagnosis continues alongside a screening campaign.
   Restricting it to the comparator would understate screening-arm costs.

Both readings are the ones that follow from the model's structure, but they are assumptions
about the costing study rather than facts from the spreadsheet. Confirm before use.
"""

from tbcore.economics import contract
from tbcore.economics.costs import CostComponent, CostStructure, cost_items_from_spreadsheet
from tbcore.economics.inputs import UncertainQuantity, sd_from_ci

from projects.vietnam_screening.config import ISO3

#: Disability weight for symptomatic TB and TB on treatment (GBD), from the DALY rows of the
#: spreadsheet. Stored separately because that row's value is prose rather than a number.
YLD_WEIGHT = UncertainQuantity(
    name="yld_weight_symptomatic_tb",
    mean=0.333,
    sd=sd_from_ci(0.224, 0.454),
    kind="beta",
)

#: Spreadsheet keys, as ``(parameter, applies_to)``.
SCREENING_COST_KEY = "Screening, per person screened"
DIAGNOSIS_ACTIVE_KEY = "Diagnosis, per positive case"
DIAGNOSIS_PASSIVE_KEY = "Diagnosis, per diagnosed case"
TREATMENT_KEY = "Treatment, per person treated"
PATIENT_COST_KEY = "Total patient cost, per person treated"

#: Which spreadsheet screening-cost row belongs to which algorithm.
SCREENING_COST_ROWS = {
    "acf_xpert": "ACF_Xpert scenario (everyone except those in treatment)",
    "acf_cxr_xpert": "ACF_CXR_Xpert scenario (everyone except those in treatment)",
    "acf_pluslife": "ACF_Pluslife scenario (everyone except those in treatment)",
    "acf_cxr_pluslife": "ACF_CXR_Pluslife scenario (everyone except those in treatment)",
}

#: Which diagnosis-cost row belongs to which algorithm. Xpert-based and Pluslife-based
#: algorithms share a row each.
DIAGNOSIS_COST_ROWS = {
    "acf_xpert": "ACF_Xpert scenario, ACF_CXR_Xpert scenario",
    "acf_cxr_xpert": "ACF_Xpert scenario, ACF_CXR_Xpert scenario",
    "acf_pluslife": "ACF_Pluslife scenario, ACF_CXR_Pluslife scenario",
    "acf_cxr_pluslife": "ACF_Pluslife scenario, ACF_CXR_Pluslife scenario",
}

PASSIVE_DIAGNOSIS_ROW = "No screening scenario"
TREATMENT_ROW = "All scenarios"
PATIENT_COST_ACTIVE_ROW = "ACF scenarios"
PATIENT_COST_PASSIVE_ROW = "No screening scenario"


def _lookup(items: dict, parameter: str, applies_to: str) -> UncertainQuantity:
    key = (parameter, applies_to)
    if key not in items:
        raise KeyError(
            f"Economic input spreadsheet has no row for {key}. Available rows: "
            f"{sorted(items)}"
        )
    return items[key]


def routine_care_components(items: dict) -> list:
    """Costs incurred regardless of whether a screening campaign runs."""
    return [
        CostComponent(
            name="passive_diagnosis",
            unit_cost=_lookup(items, DIAGNOSIS_PASSIVE_KEY, PASSIVE_DIAGNOSIS_ROW),
            quantity_output=contract.TREATMENT_STARTS_PASSIVE,
            perspective="provider",
        ),
        CostComponent(
            name="treatment_passive",
            unit_cost=_lookup(items, TREATMENT_KEY, TREATMENT_ROW),
            quantity_output=contract.TREATMENT_STARTS_PASSIVE,
            perspective="provider",
        ),
        CostComponent(
            name="patient_cost_passive",
            unit_cost=_lookup(items, PATIENT_COST_KEY, PATIENT_COST_PASSIVE_ROW),
            quantity_output=contract.TREATMENT_STARTS_PASSIVE,
            perspective="patient",
        ),
    ]


def baseline_cost_structure(items: dict = None) -> CostStructure:
    """No-screening comparator: routine care only."""
    items = items if items is not None else cost_items_from_spreadsheet(ISO3)
    return CostStructure(scenario="baseline", components=routine_care_components(items))


def screening_cost_structure(algorithm: str, items: dict = None) -> CostStructure:
    """Routine care plus the costs of a screening campaign."""
    if algorithm not in SCREENING_COST_ROWS:
        raise ValueError(
            f"No costing defined for algorithm '{algorithm}'; "
            f"known: {sorted(SCREENING_COST_ROWS)}"
        )
    items = items if items is not None else cost_items_from_spreadsheet(ISO3)

    components = routine_care_components(items) + [
        CostComponent(
            name="screening",
            unit_cost=_lookup(items, SCREENING_COST_KEY, SCREENING_COST_ROWS[algorithm]),
            quantity_output=f"n_screened_{algorithm}",
            perspective="provider",
        ),
        CostComponent(
            name="active_diagnosis",
            unit_cost=_lookup(items, DIAGNOSIS_ACTIVE_KEY, DIAGNOSIS_COST_ROWS[algorithm]),
            quantity_output=f"n_screen_positive_{algorithm}",
            perspective="provider",
        ),
        CostComponent(
            name="treatment_active",
            unit_cost=_lookup(items, TREATMENT_KEY, TREATMENT_ROW),
            quantity_output=contract.TREATMENT_STARTS_ACTIVE,
            perspective="provider",
        ),
        CostComponent(
            name="patient_cost_active",
            unit_cost=_lookup(items, PATIENT_COST_KEY, PATIENT_COST_ACTIVE_ROW),
            quantity_output=contract.TREATMENT_STARTS_ACTIVE,
            perspective="patient",
        ),
    ]
    return CostStructure(scenario=algorithm, components=components)
