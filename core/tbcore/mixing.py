"""Age mixing.

Currently homogeneous: every age group contacts every other at the same per-capita rate,
which is ``summer2``'s behaviour when no mixing matrix is set.

Heterogeneous mixing is deliberately **not** ported yet. The published Kiribati model built
a mixing matrix from three calibrated parameters (background mixing, assortativity spread,
parent-child mixing strength) anchored to an empirical ``conmat`` matrix, with the
parent-child component derived from age-specific fertility. Both the fertility series and
the ``conmat`` matrix exist for KIR only, so heterogeneous mixing cannot presently be used
for VNM, PHL or IDN at all.

Switching it on therefore needs two things, in order: the country inputs, then the port of
``age_mixing.py`` and the fertility lookups from ``kiribati_tb_modelling``. Until then this
module fails loudly rather than silently returning homogeneous mixing when heterogeneous
mixing was requested.
"""

MIXING_DATA_REQUIRED = (
    "Heterogeneous age mixing requires a country-specific contact matrix "
    "(data/shared/conmat_all_<ISO3>.csv) and an age-specific fertility series "
    "(data/<iso3>/un_fertility_rates_<ISO3>.csv). Only KIR currently has either."
)


def get_age_mixing_matrix(iso3: str, age_groups: list[str], heterogeneous: bool):
    """Mixing matrix for the age stratification, or ``None`` for homogeneous mixing.

    Parameters
    ----------
    heterogeneous
        When ``False``, returns ``None`` and ``summer2`` applies homogeneous mixing.

    Raises
    ------
    NotImplementedError
        If heterogeneous mixing is requested. Returning homogeneous mixing instead would
        silently change the transmission assumption, which is a scientific decision rather
        than an implementation detail.
    """
    if not heterogeneous:
        return None

    raise NotImplementedError(
        f"Heterogeneous mixing is not yet available for '{iso3}'. {MIXING_DATA_REQUIRED} "
        "The implementation to port is age_mixing.py in kiribati_tb_modelling. "
        "Set heterogeneous_mixing=False to proceed with homogeneous mixing."
    )
