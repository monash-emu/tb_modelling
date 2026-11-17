# Progress

Handover document. Update at the end of substantial working sessions: overwrite sections,
don't append a diary.

Last updated: 2026-11-11

## Current status

The core is built and tested: `pixi run test` is green (99 tests). The pipeline runs end to
end — build, calibrate, run scenarios, store with provenance, reload, cost — demonstrated on
Kiribati inputs by `projects/vietnam_screening/notebooks/00_smoke_test.py`.

**No analysis is valid yet.** Vietnam has no calibration targets and no parameter file, and
the model has not been checked against published Kiribati results.

### Core (`core/tbcore/`)

| Module | Status |
| --- | --- |
| `paths`, `compartments`, `config` | Done. Config is a frozen dataclass with validation and a hash used to key posteriors |
| `stratifications/` | Done. Age always applied; reachability and income optional and pluggable |
| `demography` | Done. UN population and mortality, covering IDN/KIR/PHL/VNM |
| `mixing` | **Homogeneous only.** Heterogeneous raises rather than silently downgrading |
| `interventions` | Done. Screening tools, programmes, programmatic levers, scenarios |
| `model` | Done. Births and deaths generalised over arbitrary stratifications |
| `outputs/` | Done. Names generated from active stratifications; three output sets |
| `economics/` | Done. Discounting, PSA inputs, DALYs, costing, CEA |
| `calibration` | Done. Targets supplied by projects, never hard-coded here |
| `runs` | Done. Manifest with git commit, dirty flag, config hash; posterior reuse |
| `hpc` | Done. SLURM array size derived from the parameter grid |

### Vietnam project (`projects/vietnam_screening/`)

Config, scenarios and cost structures are written and tested. Targets deliberately raise.

## Decisions made

- **Monorepo**, not branches and not one repo per project. `core/` never imports from
  `projects/`, so extraction later remains possible.
- **Vietnam at national scale**, no population scaling.
- **Homogeneous mixing by age** for now (user instruction, 2026-11-11). This also sidesteps
  the missing Vietnam contact matrix and fertility data.
- **Reachability retained but optional** — off for Vietnam and the disruptions work, on for
  the Kiribati follow-up.
- **Economics as post-processing** on saved derived outputs.
- **Screening reach into a stratum belongs to the programme**, not the stratification, so
  that outreach to people missed by a previous round is a scenario rather than a code change.
- **Programmatic levers are one mechanism** serving funding disruption (degradation) and
  enhanced clinical vigilance (improvement).
- **Posterior subsampling is seeded** (`AnalysisConfig.random_seed`), so re-running against a
  stored posterior reproduces published numbers exactly.
- **Technical appendix stays LaTeX + Make**; Quarto for executable results reports only.

## Issues found and fixed during the port

- **Screening coverage could diverge.** `-log(1-c)/d` with the published Kiribati
  configuration evaluated at 99.4% coverage of the reachable population, within half a
  percentage point of a divergent rate, with no guard. Now rejected at construction where
  the value is static.
- **People screened versus screening encounters.** The detection hazard is memoryless, so
  rate times population counts repeat screening of the same person and exceeded total
  population. Unit costs are per person screened, so resource use now uses campaign
  throughput. The two differ by `-log(1-c)/c`, which is 2.2 at 85% coverage. Removing the
  discrepancy properly needs a screened/not-yet-screened dimension.
- **Unseeded posterior subsampling** meant re-running an analysis gave slightly different
  numbers. Now seeded.
- **pytensor fails to compile** against current macOS SDK headers with the pinned pymc
  5.2.0. `PYTENSOR_FLAGS="cxx="` is set for osx-arm64 in `pixi.toml`; sampling is correct but
  slow locally. Linux, and therefore MASSIVE, is unaffected.

## Known limitation: local calibration is impractically slow

The pytensor Python fallback above costs roughly 18 seconds per draw: a 300-draw smoke chain
took 91 minutes. For comparison, the published Kiribati runs achieved about 0.16 seconds per
draw on MASSIVE with a larger model — some two orders of magnitude faster.

Consequences and options:

- Do local work with `TEST_ANALYSIS_CONFIG` or a stored posterior only; calibrate on MASSIVE.
- Posterior reuse makes this workable: re-running the smoke test against a stored posterior
  skipped calibration entirely and completed the scenarios in 10 seconds.
- Alternatively raise the `pymc` pin. 5.2.0 was inherited from the Kiribati environment; a
  newer pytensor would compile against the current SDK. That is a reproducibility decision
  rather than a maintenance one, so it has not been changed.

## Note on provenance

`git_commit` records `unknown` until the repository has its first commit, because an unborn
HEAD has no commit to read. It will populate normally thereafter. Nothing has been committed
yet.

## Blocked — waiting on input

- **Vietnam calibration targets**: which data sources are available? See
  `projects/vietnam_screening/targets.py` for the output each would bind to.
- **`data/vnm/parameters.xlsx`** does not exist. Natural history may transfer from Kiribati;
  programmatic parameters must not.
- **Cost interpretation**: two assumptions bridge the spreadsheet's scenario labels and the
  model's detection-route split. See the note at the top of
  `projects/vietnam_screening/economics.py`. Both materially change the result.
- **`cxr_positivity_prop`**: X-ray positivity among those screened, needed to count
  confirmatory tests in the triage arms.
- **Economic scope**: societal perspective assumed; discount reference year, cost horizon and
  cost-effectiveness threshold unset.
- **Pluslife sensitivities** are placeholders flagged `skip` in the inherited parameter file.
- **Contact matrices and fertility** for VNM/PHL/IDN, if heterogeneous mixing is wanted later.
- **Catastrophic costs** need a distribution of patient costs against household income.

## Open design decisions — Kiribati follow-up

Raised by the ARC proposal. None blocks the core; all block the follow-up scenarios. See
`projects/kiribati_followup/CLAUDE.md`.

1. **Household contact investigation** — no household structure. Notification-proportional
   intervention, or an explicit contact stratification?
2. **Hotspot-focused screening** — needs a spatial or risk-concentration dimension. Note that
   "hotspot" implies assortative mixing, which homogeneous between-strata mixing understates.
3. **Repeat infection testing** — needs tracking of prior screening and preventive-treatment
   history, likely a `tpt_history` dimension. Would also resolve the screened-versus-
   encounters discrepancy above.
4. **Threshold-triggered sequencing** — state-dependent switching is a technical risk under
   JAX tracing. Fallback is segmented simulation.

## Next steps

1. Reproduce published Kiribati results, to validate the port. Requires heterogeneous mixing,
   so it depends on porting `age_mixing.py` — currently deferred.
2. Shared LaTeX appendix sections in `docs/appendix_common/`, with the generated parameter
   table script.
3. Vietnam data: parameters and targets, once the questions above are answered.
4. Funding-disruptions project: income stratification parameters and equity outputs.
