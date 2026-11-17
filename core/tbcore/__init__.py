"""Project-agnostic TB transmission modelling core.

Nothing here may import from ``projects/``: the dependency runs one way only, so the core
can be extracted into its own repository if per-project access constraints ever require it.
"""

from tbcore import calibration, compartments, demography, economics, interventions, outputs, runs
from tbcore.config import (
    DEFAULT_AGE_GROUPS,
    KIRIBATI_AGE_GROUPS,
    TEST_ANALYSIS_CONFIG,
    AnalysisConfig,
    ModelConfig,
)
from tbcore.interventions import (
    ProgrammaticLever,
    Scenario,
    ScreeningProgram,
    ScreeningTool,
    ScreeningTools,
)
from tbcore.model import get_tb_model
from tbcore.parameters import get_parameters_and_priors
from tbcore.stratifications import IncomeStratification, ReachabilityStratification

__all__ = [
    "calibration",
    "compartments",
    "demography",
    "economics",
    "interventions",
    "outputs",
    "runs",
    "AnalysisConfig",
    "ModelConfig",
    "DEFAULT_AGE_GROUPS",
    "KIRIBATI_AGE_GROUPS",
    "TEST_ANALYSIS_CONFIG",
    "ProgrammaticLever",
    "Scenario",
    "ScreeningProgram",
    "ScreeningTool",
    "ScreeningTools",
    "get_tb_model",
    "get_parameters_and_priors",
    "IncomeStratification",
    "ReachabilityStratification",
]
