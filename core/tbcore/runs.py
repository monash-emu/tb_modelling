"""Run storage and provenance.

A result without provenance is not a result: months later, during peer review, the only way
to answer "which code produced this figure" is to have recorded it at the time. Every run
therefore stores the git commit, whether the working tree was dirty, the fully resolved
configuration, and a configuration hash.

The configuration hash is what makes posterior reuse safe. Calibration is expensive and
scenario runs are cheap, so posteriors are stored artefacts to be reused — but only by a
model built from an identical configuration. :func:`find_reusable_posterior` enforces that
rather than leaving it to memory.

Layout::

    <OUTPUT_ROOT>/<project>/<run_id>/
        manifest.yaml
        config.yaml
        idata.nc                      posterior, if the run calibrated
        derived/<scenario>.parquet    full derived outputs per scenario
        quantiles/<scenario>.parquet  uncertainty quantiles per scenario
"""

import platform
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

from tbcore.paths import REPO_ROOT, project_outputs

MANIFEST_FILE = "manifest.yaml"
CONFIG_FILE = "config.yaml"
POSTERIOR_FILE = "idata.nc"
DERIVED_DIR = "derived"
QUANTILES_DIR = "quantiles"


@dataclass
class RunManifest:
    """Provenance for a single run."""

    run_id: str
    project: str
    created: str
    config_hash: str
    git_commit: str = "unknown"
    git_dirty: bool = False
    description: str = ""
    has_posterior: bool = False
    scenarios: list = field(default_factory=list)
    timings: dict = field(default_factory=dict)
    python_version: str = field(default_factory=platform.python_version)

    def to_dict(self) -> dict:
        return asdict(self)


def git_state() -> tuple:
    """Current commit and whether the working tree has uncommitted changes.

    Failures are tolerated — a run on a machine without git metadata is still worth saving,
    it is just less traceable — but they are recorded as ``"unknown"`` rather than silently
    omitted.
    """
    try:
        from git import InvalidGitRepositoryError, Repo

        repo = Repo(REPO_ROOT, search_parent_directories=True)
        return repo.head.commit.hexsha, repo.is_dirty()
    except Exception:
        return "unknown", False


def new_run_id(name: str, timestamp: datetime = None) -> str:
    """Timestamped run identifier, sortable and readable."""
    stamp = (timestamp or datetime.now(timezone.utc)).strftime("%Y%m%d-%H%M%S")
    slug = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
    return f"{stamp}_{slug}"


def run_dir(project: str, run_id: str, create: bool = False) -> Path:
    path = project_outputs(project) / run_id
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def save_run(
    project: str,
    run_id: str,
    config,
    analysis_config=None,
    idata=None,
    derived_outputs: dict = None,
    quantiles: dict = None,
    description: str = "",
    timings: dict = None,
) -> Path:
    """Write a run and its provenance to disk.

    Parameters
    ----------
    derived_outputs, quantiles
        Scenario name to dataframe. Saved as parquet.
    idata
        ``arviz`` inference data, saved as netCDF when present.

    Returns
    -------
    Path
        The run directory.
    """
    path = run_dir(project, run_id, create=True)
    commit, dirty = git_state()

    if idata is not None:
        import arviz as az

        az.to_netcdf(idata, path / POSTERIOR_FILE)

    for subdir, frames in [(DERIVED_DIR, derived_outputs), (QUANTILES_DIR, quantiles)]:
        if not frames:
            continue
        (path / subdir).mkdir(exist_ok=True)
        for scenario, frame in frames.items():
            _to_parquet(frame, path / subdir / f"{scenario}.parquet")

    manifest = RunManifest(
        run_id=run_id,
        project=project,
        created=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        config_hash=config.config_hash(),
        git_commit=commit,
        git_dirty=dirty,
        description=description,
        has_posterior=idata is not None,
        scenarios=sorted(derived_outputs or quantiles or {}),
        timings=timings or {},
    )
    _write_yaml(path / MANIFEST_FILE, manifest.to_dict())

    stored_config = {"model": config.to_dict()}
    if analysis_config is not None:
        stored_config["analysis"] = analysis_config.to_dict()
    _write_yaml(path / CONFIG_FILE, stored_config)

    return path


def _to_parquet(frame: pd.DataFrame, path: Path) -> None:
    """Write a dataframe to parquet, coercing what parquet cannot represent.

    Quantile dataframes carry float column levels and occasional JAX scalars, neither of
    which parquet accepts.
    """
    frame = frame.copy()
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = pd.MultiIndex.from_tuples(
            [tuple(str(part) for part in col) for col in frame.columns]
        )
    else:
        frame.columns = [str(col) for col in frame.columns]

    for col in frame.columns:
        if frame[col].dtype == object:
            frame[col] = frame[col].map(_to_scalar)
    frame.to_parquet(path)


def _to_scalar(value):
    for attr in ("item", "tolist"):
        if hasattr(value, attr):
            try:
                return getattr(value, attr)()
            except Exception:
                return value
    return value


@dataclass
class StoredRun:
    """A run loaded back from disk."""

    path: Path
    manifest: RunManifest
    config: dict

    def derived(self, scenario: str) -> pd.DataFrame:
        return self._read(DERIVED_DIR, scenario)

    def quantiles(self, scenario: str) -> pd.DataFrame:
        return self._read(QUANTILES_DIR, scenario)

    def posterior(self):
        import arviz as az

        path = self.path / POSTERIOR_FILE
        if not path.exists():
            raise FileNotFoundError(f"Run {self.manifest.run_id} stored no posterior")
        return az.from_netcdf(path)

    def _read(self, subdir: str, scenario: str) -> pd.DataFrame:
        path = self.path / subdir / f"{scenario}.parquet"
        if not path.exists():
            available = sorted(p.stem for p in (self.path / subdir).glob("*.parquet"))
            raise FileNotFoundError(
                f"No {subdir} output for scenario '{scenario}' in run "
                f"{self.manifest.run_id}. Available: {available}"
            )
        return pd.read_parquet(path)


def load_run(project: str, run_id: str) -> StoredRun:
    path = run_dir(project, run_id)
    if not path.exists():
        raise FileNotFoundError(
            f"No run '{run_id}' for project '{project}'. "
            f"Available: {[r.run_id for r in list_runs(project)]}"
        )
    manifest = RunManifest(**_read_yaml(path / MANIFEST_FILE))
    return StoredRun(path=path, manifest=manifest, config=_read_yaml(path / CONFIG_FILE))


def list_runs(project: str) -> list:
    """Manifests of every stored run for a project, newest first."""
    root = project_outputs(project)
    manifests = []
    for path in sorted(root.glob(f"*/{MANIFEST_FILE}"), reverse=True):
        try:
            manifests.append(RunManifest(**_read_yaml(path)))
        except Exception:
            continue  # a partially written run should not break listing
    return manifests


def find_reusable_posterior(project: str, config) -> RunManifest:
    """Most recent stored posterior built from an identical model configuration.

    Returns ``None`` when there is none, which means calibration has to run.
    """
    target = config.config_hash()
    for manifest in list_runs(project):
        if manifest.has_posterior and manifest.config_hash == target:
            return manifest
    return None


def _write_yaml(path: Path, data: dict) -> None:
    with open(path, "w") as handle:
        yaml.safe_dump(data, handle, default_flow_style=False, sort_keys=False)


def _read_yaml(path: Path) -> dict:
    with open(path) as handle:
        return yaml.safe_load(handle)
