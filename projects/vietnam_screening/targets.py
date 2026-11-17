"""Calibration targets for Vietnam.

Deliberately empty. Targets are scientific inputs, not boilerplate: inventing plausible
values to make the pipeline run would produce a model that calibrates cleanly to fiction.

What is needed, and the model output each would bind to:

================================  ==========================================
Data source                       Model output
================================  ==========================================
National TB prevalence survey     ``tb_prevalence_per100k``
Case notifications by year        ``notifications``
Prevalence survey age breakdown   ``tb_prevalenceXage_{age}_per100k``
Symptom status of prevalent TB    ``perc_prev_subclinical``
Bacteriological status            ``perc_prev_infectious``
TB infection survey, if any       ``tst_posXage_{age}_perc``
================================  ==========================================

Treatment success by year is a time-variant *parameter* rather than a target, and belongs in
``data/vnm/parameters.xlsx``.
"""

MISSING_TARGETS_MESSAGE = (
    "No calibration targets are defined for Vietnam. Calibration cannot proceed until the "
    "available data sources are confirmed — see the table in this module's docstring. "
    "Model structure and the economic layer can be exercised without them using "
    "notebooks/00_smoke_test.ipynb, which runs on Kiribati inputs."
)


def get_targets() -> list:
    raise NotImplementedError(MISSING_TARGETS_MESSAGE)
