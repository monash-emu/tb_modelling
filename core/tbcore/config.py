"""Model and analysis configuration.

Configuration is a dataclass rather than a dict so that a typo is an error at construction
rather than a silently-ignored key — the published model used ``model_config.get(...)``
in places, where a mistyped key would fall back to a default without complaint.
"""

import hashlib
import json
from dataclasses import asdict, dataclass, field, replace

from tbcore.stratifications.base import StratificationModule

#: Age breaks used by the published Kiribati model. Other projects generally want fewer, but
#: see ``tbcore.stratifications.age.REQUIRED_AGE_BREAKS`` for the minimum, and note that
#: coarse bands reduce the resolution of age-weighted DALY calculations unless the
#: single-age population weighting in ``tbcore.economics.dalys`` is used.
KIRIBATI_AGE_GROUPS = ["0", "3", "5", "10", "15", "18", "40", "65"]

#: Default for new projects: enough resolution for TB epidemiology and DALY weighting
#: without the screening-specific breaks the PEARL algorithm needed.
DEFAULT_AGE_GROUPS = ["0", "5", "15", "35", "55", "70"]


@dataclass(frozen=True)
class ModelConfig:
    """Everything needed to build a model, independent of parameter values.

    Parameters
    ----------
    iso3
        Country code; selects UN demography and the contact matrix.
    start_time
        Simulation start. The long run-in before the first calibration data lets the age
        distribution and the epidemic reach equilibrium, so results are insensitive to the
        seed.
    scenario_start_time
        Year from which cumulative outputs accumulate and from which scenario levers ramp in.
        Cumulative burden is only meaningful relative to this point.
    pop_scaling
        Multiplier on national UN population, for subnational models. 1.0 for national runs.
    stratifications
        Non-age stratifications to apply, in order. Age is always applied first.
    heterogeneous_mixing
        Homogeneous by default. Heterogeneous mixing needs country-specific contact and
        fertility data that currently exists for KIR only — see :mod:`tbcore.mixing`.
    age_aggregates
        Extra reporting age bands built from model age groups, as
        ``(("3_9", ("3", "5")), ("15+", ("15", "18", "40", "65")))``. Used where calibration
        targets are reported for bands that do not match the model's own.
    output_set
        Which derived outputs to request: ``"core"`` for burden and cascade, ``"economic"``
        to add resource use, ``"full"`` for everything including per-stratum detail. Fewer
        outputs means faster runs.
    """

    iso3: str
    start_time: int = 1850
    end_time: int = 2035
    scenario_start_time: int = 2026
    seed: float = 100.0
    age_groups: tuple = tuple(DEFAULT_AGE_GROUPS)
    pop_scaling: float = 1.0
    heterogeneous_mixing: bool = False
    stratifications: tuple = ()
    output_set: str = "core"
    age_aggregates: tuple = ()

    def __post_init__(self):
        if self.end_time <= self.start_time:
            raise ValueError("end_time must be after start_time")
        if not (self.start_time <= self.scenario_start_time <= self.end_time):
            raise ValueError("scenario_start_time must fall within the simulation period")
        for strat in self.stratifications:
            if not isinstance(strat, StratificationModule):
                raise TypeError(f"{strat!r} is not a StratificationModule")
        names = [s.name for s in self.stratifications]
        if len(names) != len(set(names)):
            raise ValueError(f"Duplicate stratification names: {names}")

    @property
    def strat_names(self) -> list[str]:
        return [s.name for s in self.stratifications]

    def with_changes(self, **kwargs) -> "ModelConfig":
        """Return a copy with fields replaced, e.g. for sensitivity analyses."""
        return replace(self, **kwargs)

    def to_dict(self) -> dict:
        """Plain-data representation, for saving alongside run outputs."""
        data = asdict(self)
        data["age_groups"] = list(self.age_groups)
        data["stratifications"] = [
            {"name": s.name, "strata": list(s.strata), "class": type(s).__name__}
            for s in self.stratifications
        ]
        return data

    def config_hash(self) -> str:
        """Short stable hash of the configuration.

        Used to key stored posteriors: a posterior may only be reused for a model built from
        an identical configuration.
        """
        blob = json.dumps(self.to_dict(), sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:12]


@dataclass
class AnalysisConfig:
    """Calibration and scenario-run settings.

    ``random_seed`` fixes the subsampling of posterior draws used for scenario runs. It
    defaults to a fixed value rather than ``None`` so that re-running an analysis against a
    stored posterior reproduces the published numbers exactly.
    """

    chains: int = 8
    cores: int = 8
    tune: int = 5000
    draws: int = 10000
    burn_in: int = 5000
    full_runs_samples: int = 1000
    sampler: str = "DEMetropolisZ"
    random_seed: int = 0
    scenarios: list = field(default_factory=list)

    def __post_init__(self):
        if self.burn_in >= self.tune + self.draws:
            raise ValueError("burn_in must be smaller than the total chain length")
        if self.full_runs_samples > self.draws - self.burn_in:
            raise ValueError(
                f"Requested {self.full_runs_samples} full-run samples but only "
                f"{self.draws - self.burn_in} post-burn-in draws are available."
            )

    def to_dict(self) -> dict:
        data = asdict(self)
        data["scenarios"] = [getattr(s, "sc_id", str(s)) for s in self.scenarios]
        return data


#: Fast settings for smoke tests. Results are meaningless; the point is that the pipeline
#: runs end to end in seconds.
TEST_ANALYSIS_CONFIG = AnalysisConfig(
    chains=2, cores=2, tune=50, draws=100, burn_in=50, full_runs_samples=50
)
