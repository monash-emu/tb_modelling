"""Screening-reachability stratification.

Retained from the published Kiribati model, where roughly 15% of the population did not
participate in active screening. Hard-to-reach individuals receive no active screening by
default, have reduced access to passive case detection, and carry a higher risk of
acquiring infection.

Off by default. Switched on for the Kiribati follow-up study, which must reproduce the
published model and additionally evaluate strategies that deliberately target the people
PEARL missed. Those strategies are expressed by a screening programme declaring its own
``reachability`` multipliers, which override the default of no reach — see
``default_screening_multipliers`` below. Mixing between strata is assumed homogeneous.
"""

from summer2 import CompartmentalModel
from summer2.parameters import Parameter

from tbcore.compartments import COMPARTMENTS
from tbcore.stratifications.base import BuildContext, StratificationModule

REACHABLE = "reachable"
UNREACHABLE = "unreachable"


class ReachabilityStratification(StratificationModule):

    name = "reachability"
    strata = [REACHABLE, UNREACHABLE]

    #: Routine screening rounds reach only the reachable stratum — that is the definition of
    #: the stratum. A dedicated outreach programme overrides this by declaring its own
    #: multipliers, rather than by changing the stratification.
    default_screening_multipliers = {REACHABLE: 1.0, UNREACHABLE: 0.0}

    @property
    def population_split(self) -> dict:
        reachable_frac = Parameter("reachable_pop_frac")
        return {REACHABLE: reachable_frac, UNREACHABLE: 1.0 - reachable_frac}

    def apply(self, model: CompartmentalModel, context: BuildContext) -> None:
        strat = self._base_stratification(COMPARTMENTS)

        # Higher susceptibility to infection among the hard-to-reach.
        for flow in context.flows.infection:
            strat.set_flow_adjustments(
                flow_name=flow,
                adjustments={REACHABLE: 1.0, UNREACHABLE: Parameter("rel_sus_unreachable")},
            )

        # Reduced access to routine diagnosis among the hard-to-reach.
        for flow in context.flows.passive_detection:
            strat.set_flow_adjustments(
                flow_name=flow,
                adjustments={
                    REACHABLE: 1.0,
                    UNREACHABLE: Parameter("rel_detection_unreachable"),
                },
            )

        self._apply_screening_adjustments(strat, context)

        model.stratify_with(strat)
