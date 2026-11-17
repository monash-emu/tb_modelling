# TB modelling monorepo — repository instructions

Shared core model plus one directory per study. Read
`~/Documents/agentic_code/CLAUDE.md` for the modelling conventions that apply across
projects, and the `CLAUDE.md` inside a project directory before working on that project.

## Layout

| Path | Contents |
| --- | --- |
| `core/tbcore/` | Project-agnostic model, outputs, economics, calibration, run registry |
| `data/shared/` | UN demography and contact matrices (multi-country) |
| `data/<iso3>/` | Country-specific targets, notifications, parameters |
| `projects/<name>/` | Per-study config, scenarios, targets, notebooks, appendix, HPC |
| `docs/appendix_common/` | LaTeX sections shared between project appendices |
| `tests/` | Regression and structural tests for the core |

`core/tbcore` must never import from `projects/`. The dependency runs one way only, so the
core can be extracted into its own repository later without rework.

## Environment

pixi, one environment for the whole repo:

```bash
pixi install
pixi run lab          # jupyter lab
pixi run test         # pytest
```

Never `pip install` into the environment directly; add dependencies to `pixi.toml`.

## Non-negotiables

- **Stratifications are optional and pluggable.** Reachability is off by default and on only
  for the Kiribati follow-up. Never hard-code a stratification name into an output request —
  build output names from the active stratifications.
- **Economics is post-processing.** Cost and DALY code operates on saved derived-output
  dataframes. Nothing economic goes inside the `summer2` model.
- **Outputs are written through `tbcore.runs`**, which records the git commit, the resolved
  config and a config hash alongside every run. Results without provenance are not results.
- **Posteriors are reusable artefacts.** Changing scenarios or cost assumptions must not
  require re-running MCMC.
- Model structure changes require the corresponding appendix update in the same commit.

## Current status

See `PROGRESS.md` for what is built, what is in flight and what is blocked.
