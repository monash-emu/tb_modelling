"""Screening tools, screening programmes, programmatic levers and scenarios.

A **screening tool** is a diagnostic pathway: which states it detects and with what
sensitivity, where detected people go, and how many tests of each type it consumes.

A **screening programme** applies a tool to a target population over a time window at a
given coverage.

A **programmatic lever** perturbs routine services rather than adding a campaign. The same
mechanism serves opposite purposes across projects: degraded detection and treatment success
for the funding-disruptions study, improved detection for the "community awareness and
clinical vigilance" strategy in the Kiribati follow-up.

A **scenario** combines these with any parameter overrides.
"""

from dataclasses import dataclass, field

import jax.numpy as jnp
from summer2.functions import time as stf
from summer2.parameters import Function, Parameter

from tbcore.compartments import ACTIVE_COMPS

#: Coverage above this fraction of the targeted population is rejected. The screening rate
#: is derived as ``-log(1 - coverage) / duration``, which diverges at 1. The published
#: Kiribati configuration ran at 0.845 / 0.85 = 99.4% of the reachable population, within
#: half a percentage point of that divergence, and the original code had no guard.
MAX_COVERAGE_PROP = 0.999


@dataclass(frozen=True)
class ScreeningTool:
    """A diagnostic pathway applied during active screening.

    Parameters
    ----------
    name
        Identifier used in flow and output names.
    sensitivities
        Source compartment to probability of being detected by this pathway.
    dest_comp
        Where detected people go: ``"treatment"`` for disease screening, ``"cleared"`` for
        infection screening followed by preventive treatment.
    success_prop
        Probability of completing the resulting treatment, given a positive screen.
    tests_per_person
        Test type to number of tests consumed per person screened. Values may be
        ``Parameter`` objects where the count depends on an upstream triage step, for
        example confirmatory Xpert performed only on chest X-ray positives. Required for
        resource-use outputs and hence for costing.
    """

    name: str
    sensitivities: dict
    dest_comp: str
    success_prop: object = 1.0
    tests_per_person: dict = field(default_factory=dict)

    def __getitem__(self, key):
        # Compatibility with the published model, which passed tools around as plain dicts.
        return getattr(self, key)


def disease_screening_tool(name: str, test: str, tests_per_person: dict = None) -> ScreeningTool:
    """Tool detecting active disease, with per-state sensitivity parameters.

    Sensitivity parameters follow the naming of the parameter spreadsheet,
    ``prev_se_{compartment}_{test}``.
    """
    return ScreeningTool(
        name=name,
        sensitivities={state: Parameter(f"prev_se_{state}_{test}") for state in ACTIVE_COMPS},
        dest_comp="treatment",
        success_prop=1.0,
        tests_per_person=tests_per_person or {},
    )


class ScreeningTools:
    """Tools used by the published Kiribati model, plus the generic building blocks."""

    SSX = disease_screening_tool("ssx", "ssx", {"symptom_screen": 1.0})
    CXR = disease_screening_tool("cxr", "cxr", {"symptom_screen": 1.0, "cxr": 1.0})
    PLTS = disease_screening_tool(
        # FIXME: Pluslife sensitivities are placeholders flagged `skip` in the Kiribati
        # parameter file. They must be replaced before any Pluslife scenario is interpreted.
        "plts",
        "plts",
        {"symptom_screen": 1.0, "cxr": 1.0, "plts": 1.0},
    )

    #: Infection screening by tuberculin skin test, followed by preventive treatment.
    #: Detected people move to ``cleared`` rather than ``treatment``.
    TST = ScreeningTool(
        name="tst",
        sensitivities={
            "incipient": Parameter("prev_se_incipient_tst"),
            "contained": Parameter("prev_se_contained_tst"),
        },
        dest_comp="cleared",
        success_prop=Parameter("tpt_completion_perc") / 100.0,
        tests_per_person={"tst": 1.0},
    )

    #: The PEARL algorithm: frontline Xpert for 35% of those screened, chest X-ray alone for
    #: the remaining 65%. The split is a property of how the programme was delivered, not of
    #: either test, so it is baked into this composite tool.
    PEARL = ScreeningTool(
        name="pearl",
        sensitivities={
            state: 0.35 * Parameter(f"prev_se_{state}_pearl")
            + 0.65 * Parameter(f"prev_se_{state}_cxr")
            for state in ACTIVE_COMPS
        },
        dest_comp="treatment",
        success_prop=1.0,
        tests_per_person={"symptom_screen": 1.0, "cxr": 1.0, "xpert": 0.35},
    )


class ScreeningProgram:
    """A screening campaign: one tool, one time window, one coverage level.

    Parameters
    ----------
    coverage_prop
        Proportion of the *targeted* population screened over the campaign. "Targeted" means
        after the strata multipliers below are applied, so a programme excluding under-tens
        covers this proportion of those aged ten and over.
    strata_coverage_multipliers
        Stratification name to ``{stratum: multiplier}``. Used for age eligibility, and for
        declaring reach into strata a stratification would otherwise exclude — which is how
        outreach to people missed by a previous round is expressed.
    population_coverage_basis
        Optional divisor converting population-wide coverage into coverage among those the
        programme can actually reach. The published Kiribati model divided by
        ``reachable_pop_frac``; supply ``Parameter("reachable_pop_frac")`` to reproduce it.
        Leave as ``None`` for the straightforward reading, in which coverage already refers
        to the targeted population.

        Be aware that this divisor is what pushed the published configuration to 99.4% and
        close to a divergent screening rate. When it is a calibrated parameter the resulting
        coverage cannot be checked at construction time, so the guard below only applies to
        fixed values.
    """

    def __init__(
        self,
        name: str,
        scr_tool: ScreeningTool,
        start_time: float,
        end_time: float,
        coverage_prop: float,
        strata_coverage_multipliers: dict = None,
        population_coverage_basis=None,
    ):
        if end_time <= start_time:
            raise ValueError(f"Programme '{name}': end_time must be after start_time")
        if not 0.0 < coverage_prop <= MAX_COVERAGE_PROP:
            raise ValueError(
                f"Programme '{name}': coverage_prop must be in (0, {MAX_COVERAGE_PROP}], "
                f"got {coverage_prop}."
            )

        self.name = name
        self.scr_tool = scr_tool
        self.start_time = start_time
        self.end_time = end_time
        self.coverage_prop = coverage_prop
        self.strata_coverage_multipliers = strata_coverage_multipliers or {}
        self.population_coverage_basis = population_coverage_basis

        self.effective_coverage = self._effective_coverage()
        self.raw_screening_func = self._build_screening_func()
        self.throughput_func = self._build_throughput_func()

    def _effective_coverage(self):
        if self.population_coverage_basis is None:
            return self.coverage_prop
        effective = self.coverage_prop / self.population_coverage_basis
        if isinstance(effective, float) and effective > MAX_COVERAGE_PROP:
            raise ValueError(
                f"Programme '{self.name}': coverage of {self.coverage_prop} over a basis of "
                f"{self.population_coverage_basis} implies {effective:.3f} coverage of the "
                f"reachable population, above the {MAX_COVERAGE_PROP} limit. The screening "
                "rate would diverge."
            )
        return effective

    def _build_screening_func(self):
        """Constant screening *hazard* over the campaign, delivering the requested coverage.

        Coverage ``c`` over a campaign of duration ``d`` implies a constant per-time hazard
        of ``-log(1 - c) / d``, since screening is a competing-risk process rather than a
        deterministic allocation. This is the rate that drives detection flows.
        """
        duration = self.end_time - self.start_time
        rate = Function(_coverage_to_rate, [self.effective_coverage, duration])
        # Step on at start_time and off at end_time, with a near-instantaneous ramp so the
        # solver sees a continuous function.
        return self._campaign_window(rate)

    def _build_throughput_func(self):
        """People screened per unit time, which is *not* the detection hazard.

        The hazard above is memoryless, so over a campaign it generates more screening
        events than there are people: an individual who screens negative remains eligible
        and can be screened again. That is the correct basis for detection, because the
        model has no screening-history dimension, but it is the wrong basis for costing.

        Unit costs are expressed per person screened, and a real campaign screens each
        participant once. The number of people screened per unit time is therefore
        ``coverage / duration`` times the eligible population, whose integral over the
        campaign is exactly ``coverage`` times that population.

        FIXME: the two differ by a factor of ``-log(1-c)/c``, which is 2.2 at 85% coverage.
        Removing the discrepancy entirely would need a screened/not-yet-screened dimension,
        which is also what evaluating repeat-screening decision rules will require.
        """
        duration = self.end_time - self.start_time
        return self._campaign_window(self.effective_coverage / duration)

    def _campaign_window(self, value):
        """Value held over the campaign window, zero outside it."""
        return stf.get_linear_interpolation_function(
            [self.start_time - 0.01, self.start_time, self.end_time - 0.01, self.end_time],
            [0.0, value, value, 0.0],
        )

    def __repr__(self) -> str:
        return (
            f"ScreeningProgram({self.name}, tool={self.scr_tool.name}, "
            f"{self.start_time}-{self.end_time}, coverage={self.coverage_prop})"
        )


def _coverage_to_rate(coverage_prop, duration):
    return -jnp.log(1.0 - coverage_prop) / duration


@dataclass(frozen=True)
class ProgrammaticLever:
    """A sustained change to routine TB services, ramping in from a start time.

    Multipliers are relative to the calibrated baseline, so 1.0 is no change, values below 1
    represent degradation (funding disruption) and values above 1 represent improvement
    (greater clinical vigilance shortening diagnosis delay).

    Parameters
    ----------
    detection_rate_rel
        Multiplier on the passive TB detection rate.
    tx_success_rel
        Multiplier on the treatment success proportion. Capped at 1.0 when applied, since a
        proportion cannot exceed one.
    transmission_rel
        Multiplier on the transmission rate, for disruption scenarios that act on contact
        patterns or infection control rather than on services.
    ramp_years
        Time over which the change phases in from the start time. A step change is rarely
        realistic and is harder on the solver.
    """

    start_time: float
    detection_rate_rel: float = 1.0
    tx_success_rel: float = 1.0
    transmission_rel: float = 1.0
    ramp_years: float = 1.0

    def __post_init__(self):
        for name in ("detection_rate_rel", "tx_success_rel", "transmission_rel"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must not be negative")
        if self.ramp_years <= 0:
            raise ValueError("ramp_years must be positive")

    def is_active(self) -> bool:
        return not (
            self.detection_rate_rel == 1.0
            and self.tx_success_rel == 1.0
            and self.transmission_rel == 1.0
        )

    def multiplier_func(self, relative_value: float):
        """Time-variant multiplier ramping from 1.0 to ``relative_value``."""
        return stf.get_linear_interpolation_function(
            [self.start_time, self.start_time + self.ramp_years], [1.0, relative_value]
        )


@dataclass
class Scenario:
    """A named combination of screening programmes, levers and parameter overrides."""

    sc_id: str
    sc_name: str
    scr_prgs: list = field(default_factory=list)
    lever: ProgrammaticLever = None
    params_ow: dict = field(default_factory=dict)
    sc_name_2: str = None
    description: str = ""

    def __post_init__(self):
        if self.sc_name_2 is None:
            self.sc_name_2 = self.sc_name
        names = [p.name for p in self.scr_prgs]
        if len(names) != len(set(names)):
            raise ValueError(f"Scenario '{self.sc_id}' has duplicate programme names: {names}")

    @property
    def tools(self) -> list[str]:
        """Distinct screening tool names used, for building resource-use outputs."""
        return sorted({p.scr_tool.name for p in self.scr_prgs})
