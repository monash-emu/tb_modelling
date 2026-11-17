# Vietnam mass TB screening — cost-effectiveness analysis

Specification section 2.1. Compares mass screening algorithms for active TB against no
screening, with costs and DALYs.

## Model

National scale, no population scaling. Homogeneous mixing by age. No reachability
stratification — the specification removes it for this application. Age groups are coarser
than Kiribati's, since the PEARL algorithm's eligibility breaks at 3, 10 and 18 are not
needed here; DALY weighting stays accurate because it is population-weighted within bands.

## Scenarios

Four active case-finding algorithms from the economic input spreadsheet, against a
no-screening baseline:

| Arm | Pathway | Status |
| --- | --- | --- |
| `acf_xpert` | Universal Xpert | Structure defined |
| `acf_cxr_xpert` | X-ray triage, Xpert confirmation | Needs `cxr_positivity_prop` |
| `acf_pluslife` | Universal tongue swab | **Placeholder sensitivities** |
| `acf_cxr_pluslife` | X-ray triage, tongue-swab confirmation | **Placeholder sensitivities** |

Provisional arms are excluded from `all_scenarios()` by default, so that a full run cannot
accidentally produce interpretable-looking results from placeholder values.

## Economics

`economics.py` maps spreadsheet rows onto model outputs. **Read the interpretation note at
the top of that module before using it**: the spreadsheet labels unit costs by scenario,
while the model produces quantities by detection route, and bridging the two requires two
assumptions that materially change the result.

Costs are post-processing on stored derived outputs, so a change of unit cost, discount rate
or perspective during peer review is a seconds-long re-run and never requires recalibration.

## Blocked

- **Calibration targets** — `targets.py` raises rather than inventing values. See its
  docstring for what is needed.
- **`data/vnm/parameters.xlsx`** — does not exist. Natural history parameters may transfer
  from Kiribati, but programmatic ones (treatment success by year, detection scale-up) must
  be Vietnam-specific.
- **`cxr_positivity_prop`** — X-ray positivity among those screened, needed to count
  confirmatory tests in the triage arms.
- **Economic scope** — societal perspective assumed; discount reference year, cost horizon
  and cost-effectiveness threshold all unset.

Until these are resolved, use `notebooks/00_smoke_test.ipynb`, which exercises the whole
pipeline on Kiribati inputs.
