"""Pluggable stratification modules.

The published Kiribati model welded its ``reachability`` stratification into the model
builder and into every derived-output request. That made "remove reachability" a rewrite
rather than a configuration change, and left no route to adding an income dimension.

Here a stratification is an object that knows its own name, strata, population split and
adjustments, and is applied to the model in order. Output requests are generated from
whichever modules are active (see ``tbcore.outputs``), so no output code names a
stratification literally.

Age is handled separately in :mod:`tbcore.stratifications.age` because it is always present,
uses ``summer2.AgeStratification`` and carries the mixing matrix.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from summer2 import CompartmentalModel, Stratification


@dataclass
class FlowRegistry:
    """Names of flows created while building the unstratified model.

    Stratification modules need these to apply adjustments: a module that modifies
    susceptibility must adjust every infection flow, and one that modifies access to care
    must adjust every passive-detection flow. Collecting the names as the model is built
    avoids each module re-deriving them and silently missing one when the structure changes.
    """

    infection: list[str] = field(default_factory=list)
    passive_detection: list[str] = field(default_factory=list)
    screening: list[str] = field(default_factory=list)
    natural_death: list[str] = field(default_factory=list)
    tb_death: list[str] = field(default_factory=list)

    def all_names(self) -> list[str]:
        return (
            self.infection
            + self.passive_detection
            + self.screening
            + self.natural_death
            + self.tb_death
        )


@dataclass
class BuildContext:
    """What a stratification module needs to know about the model being built.

    Carries the screening programmes as well as the flow inventory, because a programme may
    target specific strata of a non-age dimension. The Kiribati study assumed active
    screening never reached the hard-to-reach stratum, but the follow-up study explicitly
    compares strategies that *do* target people missed by the first round, so that
    assumption belongs to the programme, not to the stratification.
    """

    flows: FlowRegistry
    screening_programs: list = field(default_factory=list)

    def program_multipliers(self, strat_name: str, program) -> dict:
        """Per-stratum coverage multipliers a programme declares for a stratification."""
        return program.strata_coverage_multipliers.get(strat_name, {})


class StratificationModule(ABC):
    """A non-age stratification that can be switched on or off per project."""

    #: Stratification name, used as the ``X{name}_{stratum}`` segment of output names.
    name: str

    #: Stratum labels, in order.
    strata: list[str]

    #: Whether derived outputs should additionally be requested for each stratum of this
    #: dimension. Switch off for dimensions that exist to modify dynamics but are not
    #: themselves of reporting interest.
    report_by_stratum: bool = True

    #: Default coverage multiplier applied to a screening programme in each stratum, used
    #: when the programme does not declare its own. ``None`` means full coverage everywhere.
    default_screening_multipliers: dict = None

    @property
    @abstractmethod
    def population_split(self) -> dict:
        """Proportion of the population in each stratum at model start.

        Values may be ``summer2`` ``Parameter`` objects. Also used to split births, so that
        the stratum composition stays constant over time unless a module says otherwise.
        """

    @abstractmethod
    def apply(self, model: CompartmentalModel, context: BuildContext) -> None:
        """Build the ``summer2`` stratification and apply it to ``model``."""

    def _base_stratification(self, compartments: list[str]) -> Stratification:
        strat = Stratification(name=self.name, strata=self.strata, compartments=compartments)
        strat.set_population_split(proportions=self.population_split)
        return strat

    def _apply_screening_adjustments(self, strat, context: BuildContext) -> None:
        """Adjust each screening flow by the programme's reach into each stratum.

        A programme declaring multipliers for this stratification overrides the module
        default, which is how a targeted outreach round is expressed: it simply declares
        non-zero coverage in a stratum that routine rounds do not reach.
        """
        defaults = self.default_screening_multipliers or {s: 1.0 for s in self.strata}
        for program in context.screening_programs:
            declared = context.program_multipliers(self.name, program)
            unknown = set(declared) - set(self.strata)
            if unknown:
                raise ValueError(
                    f"Screening programme '{program.name}' declares '{self.name}' "
                    f"multipliers for unknown strata {sorted(unknown)}; "
                    f"valid strata are {self.strata}."
                )
            adjustments = {s: declared.get(s, defaults[s]) for s in self.strata}
            for source_comp in program.scr_tool["sensitivities"]:
                strat.set_flow_adjustments(f"{program.name}_{source_comp}", adjustments)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(strata={self.strata})"
