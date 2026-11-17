"""Model configuration for the Vietnam mass-screening cost-effectiveness analysis.

National scale, so UN demography is used directly with no population scaling.

Mixing is homogeneous by age. Heterogeneous mixing would need a Vietnam contact matrix and
fertility series, neither of which exists yet — see :mod:`tbcore.mixing`.
"""

from tbcore.config import AnalysisConfig, ModelConfig

ISO3 = "VNM"

#: Coarser than the Kiribati model, which needed breaks at 3, 10 and 18 for the PEARL
#: screening algorithm's age eligibility rules. Retains the breaks required by the
#: early-infection parameters (0, 5, 15) and enough resolution above 15 for DALY weighting,
#: which is population-weighted within each band and so stays accurate — see
#: tbcore.economics.dalys.
AGE_GROUPS = ("0", "5", "15", "35", "55", "70")

#: Screening is assumed to start in 2027 and run for one year. Provisional.
SCREENING_START = 2027
SCREENING_END = 2028

#: Analytic horizon for the cost-effectiveness analysis.
HORIZON_END = 2047


VIETNAM_CONFIG = ModelConfig(
    iso3=ISO3,
    start_time=1850,
    end_time=HORIZON_END,
    scenario_start_time=SCREENING_START,
    age_groups=AGE_GROUPS,
    pop_scaling=1.0,
    heterogeneous_mixing=False,
    # No reachability stratification: the specification removes it for this application.
    # No income stratification: that belongs to the funding-disruptions study.
    stratifications=(),
    output_set="economic",
)

ANALYSIS_CONFIG = AnalysisConfig(
    chains=8, cores=8, tune=5000, draws=10000, burn_in=5000, full_runs_samples=1000
)
