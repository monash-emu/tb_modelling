"""Filesystem locations for the repository.

Output location can be redirected with the ``TB_OUTPUT_ROOT`` environment variable, which
is how analyses are pointed at scratch storage on the HPC.
"""

import os
from pathlib import Path


def _find_repo_root() -> Path:
    """Walk up from this file to the repository root.

    Resolved from ``__file__`` rather than the working directory, so that imports behave
    identically from notebooks, scripts and HPC jobs.
    """
    for parent in Path(__file__).resolve().parents:
        if (parent / ".git").exists() or (parent / "pixi.toml").exists():
            return parent
    raise FileNotFoundError("Could not locate repository root from tbcore.paths")


REPO_ROOT = _find_repo_root()

DATA_ROOT = REPO_ROOT / "data"
SHARED_DATA = DATA_ROOT / "shared"
PROJECTS_ROOT = REPO_ROOT / "projects"
DOCS_ROOT = REPO_ROOT / "docs"
APPENDIX_COMMON = DOCS_ROOT / "appendix_common"

OUTPUT_ROOT = Path(os.environ.get("TB_OUTPUT_ROOT", REPO_ROOT / "outputs"))


def country_data(iso3: str) -> Path:
    """Directory holding country-specific inputs (targets, notifications, parameters)."""
    return DATA_ROOT / iso3.lower()


def project_path(project: str) -> Path:
    """Directory for a given study."""
    path = PROJECTS_ROOT / project
    if not path.exists():
        raise FileNotFoundError(f"No project directory at {path}")
    return path


def project_outputs(project: str) -> Path:
    """Run-output directory for a given study, created on demand."""
    path = OUTPUT_ROOT / project
    path.mkdir(parents=True, exist_ok=True)
    return path
