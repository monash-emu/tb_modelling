"""Income stratification for the funding-disruptions study.

Splits the population into lower- and higher-income groups with differential epidemiological
parameters, so that the equity consequences of funding disruption can be measured rather
than assumed. Structurally this is the same kind of dimension as reachability: a population
split carrying differential susceptibility and differential access to care.

PROVISIONAL. The strata definition, the parameters that differ between them, and whether
mixing should be assortative by income are all unresolved — see PROGRESS.md. Parameter names
below are placeholders and must be reconciled with a data source before any analysis.
"""

from summer2 import CompartmentalModel
from summer2.parameters import Parameter

from tbcore.compartments import COMPARTMENTS
from tbcore.stratifications.base import BuildContext, StratificationModule

LOWER_INCOME = "lower"
HIGHER_INCOME = "higher"


class IncomeStratification(StratificationModule):
    """Binary income stratification with differential infection risk and care access.

    Parameters
    ----------
    assortative_mixing : bool
        Placeholder for income-assortative contact patterns. Not yet implemented; mixing is
        homogeneous between income groups, matching the treatment of reachability in the
        published model. Setting this to ``True`` raises rather than silently misrepresenting
        the assumption.
    """

    name = "income"
    strata = [LOWER_INCOME, HIGHER_INCOME]

    def __init__(self, assortative_mixing: bool = False):
        if assortative_mixing:
            raise NotImplementedError(
                "Income-assortative mixing is not implemented. Mixing between income groups "
                "is currently homogeneous; implementing assortativity requires a contact "
                "data source and a decision on how it interacts with the age mixing matrix."
            )
        self.assortative_mixing = assortative_mixing

    @property
    def population_split(self) -> dict:
        lower_frac = Parameter("lower_income_pop_frac")
        return {LOWER_INCOME: lower_frac, HIGHER_INCOME: 1.0 - lower_frac}

    def apply(self, model: CompartmentalModel, context: BuildContext) -> None:
        strat = self._base_stratification(COMPARTMENTS)

        # Higher infection risk in the lower-income group (crowding, occupational exposure).
        for flow in context.flows.infection:
            strat.set_flow_adjustments(
                flow_name=flow,
                adjustments={
                    LOWER_INCOME: Parameter("rel_sus_lower_income"),
                    HIGHER_INCOME: 1.0,
                },
            )

        # Poorer access to routine diagnosis in the lower-income group. This is the channel
        # through which funding disruption is expected to widen inequity, so it must be
        # present even when the disruption scenarios act on the aggregate detection rate.
        for flow in context.flows.passive_detection:
            strat.set_flow_adjustments(
                flow_name=flow,
                adjustments={
                    LOWER_INCOME: Parameter("rel_detection_lower_income"),
                    HIGHER_INCOME: 1.0,
                },
            )

        # Screening reaches both income groups equally unless a programme says otherwise.
        self._apply_screening_adjustments(strat, context)

        model.stratify_with(strat)
