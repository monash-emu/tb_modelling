"""Compartment definitions for the TB natural history structure.

The structure follows the ICE-TB framework: active disease is cross-classified by symptom
status (``subclin`` / ``clin``) and infectiousness (``lowinf`` / ``inf``), with bidirectional
transitions along each dimension and each transition changing only one dimension at a time.

Non-disease infection uses three states rather than a single latent compartment, so that
immune containment, clearance and loss of immune control are explicit, and so that viable
infection can be distinguished from TST positivity.
"""

# "mtb_naive" rather than "susceptible": several non-naive compartments remain partially
# susceptible to reinfection, so "susceptible" would be misleading.
MTB_NAIVE = "mtb_naive"

#: Non-disease infection states.
LATENT_COMPS = ["incipient", "contained", "cleared"]

#: Active disease states, named ``{symptom_status}_{infectiousness}``.
ACTIVE_COMPS = ["subclin_lowinf", "clin_lowinf", "subclin_inf", "clin_inf"]

#: Only active disease transmits; relative infectiousness is adjusted during stratification.
INFECTIOUS_COMPS = ACTIVE_COMPS

POST_DISEASE_COMPS = ["treatment", "recovered"]

COMPARTMENTS = [MTB_NAIVE] + LATENT_COMPS + ACTIVE_COMPS + POST_DISEASE_COMPS

#: Compartments susceptible to (re)infection. Reinfection risk relative to naive is
#: controlled by ``rel_sus_*`` parameters.
SUSCEPTIBLE_COMPS = [MTB_NAIVE, "contained", "cleared", "recovered"]

#: Symptomatic disease — the only states detected by passive case finding, the only states
#: carrying TB mortality risk, and the states accruing symptomatic-TB YLDs.
SYMPTOMATIC_COMPS = [c for c in ACTIVE_COMPS if c.startswith("clin_")]

ASYMPTOMATIC_COMPS = [c for c in ACTIVE_COMPS if c.startswith("subclin_")]

#: Viable infection, i.e. excluding cleared infection. Distinct from TST positivity, which
#: includes a fraction of cleared and recovered individuals.
VIABLE_INFECTION_COMPS = ["incipient", "contained"]


def is_asymptomatic(compartment: str) -> bool:
    return compartment.startswith("subclin_")


def is_low_infectious(compartment: str) -> bool:
    return compartment.endswith("_lowinf")
