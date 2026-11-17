# TB modelling

A shared core for *Mycobacterium tuberculosis* transmission modelling, plus the applied
studies built on it. Developed at the School of Public Health and Preventive Medicine,
Monash University.

> **Status: under active development. No analysis here is validated.**
> The core is tested and the pipeline runs end to end, but the model has not yet been
> checked against published results, and no study in `projects/` has calibration data.
> See [PROGRESS.md](PROGRESS.md) for what is built, what is blocked and what is assumed.

## Background

The model structure follows the [ICE-TB framework](https://doi.org/10.1016/S1473-3099(24)00805-6):
active disease is represented along two independent dimensions — symptom status and
infectiousness — with bidirectional transitions, and each transition changing only one
dimension. Non-disease infection uses three states (incipient, contained, cleared) rather
than a single latent compartment, so immune containment, clearance and loss of control are
explicit, and viable infection is distinguished from tuberculin skin test positivity.

It derives from the model developed for the PEARL population-wide screening study in South
Tarawa, Kiribati ([monash-emu/kiribati_tb_modelling](https://github.com/monash-emu/kiribati_tb_modelling)),
generalised so that stratifications are optional, outputs are configurable, and economic
analysis is possible.

Built with [summer2](https://github.com/monash-emu/summer2) and calibrated with
[estival](https://github.com/monash-emu/estival) and [PyMC](https://www.pymc.io/).

## Layout

| Path | Contents |
| --- | --- |
| `core/tbcore/` | Project-agnostic model, outputs, economics, calibration, run registry |
| `core/tbcore/stratifications/` | Pluggable stratifications: age, reachability, income |
| `core/tbcore/economics/` | Discounting, DALYs, costing, cost-effectiveness analysis |
| `data/shared/` | UN demography, covering IDN, KIR, PHL and VNM |
| `data/<iso3>/` | Country-specific parameters, targets and economic inputs |
| `projects/<name>/` | Per-study configuration, scenarios, notebooks and appendix |
| `tests/` | Structural and regression tests for the core |

`core/tbcore` never imports from `projects/`. The dependency runs one way, so the core can
be extracted into its own package if needed.

## Studies

| Study | Question | Status |
| --- | --- | --- |
| `vietnam_screening` | Cost-effectiveness of mass TB screening algorithms | Structure complete; awaiting calibration data |
| `funding_disruptions` | Impact and equity consequences of TB funding disruption | Not started |
| `kiribati_followup` | What should follow population-wide screening in Kiribati | Requirements captured; design decisions open |

## Installation

Environments are managed with [pixi](https://pixi.sh/):

```bash
pixi install
pixi run test     # 104 tests
pixi run lab      # jupyter lab
```

## Usage

Build and run a model:

```python
from tbcore.config import ModelConfig
from tbcore.model import get_tb_model
from tbcore.parameters import get_parameters_and_priors

params, priors, tv_params = get_parameters_and_priors("kir")

config = ModelConfig(
    iso3="KIR",
    age_groups=("0", "5", "15", "35", "55", "70"),
    heterogeneous_mixing=False,
)
model = get_tb_model(config, tv_params)
model.run(parameters=params)
outputs = model.get_derived_outputs_df()
```

An end-to-end demonstration — calibration, scenarios, storage with provenance, costing and
cost-effectiveness — is in
[`projects/vietnam_screening/notebooks/00_smoke_test.py`](projects/vietnam_screening/notebooks/00_smoke_test.py).
It runs on Kiribati inputs, since Vietnam has no calibration data yet, and its numbers are
illustrative only.

## Design notes

**Stratifications are optional.** Reachability, income and any future dimension are
pluggable modules. Derived output names are generated from whichever stratifications are
active, following the convention `base_nameXstrat_stratum`, so no output code names a
stratification literally.

**Economic analysis is post-processing.** Costs and DALYs are computed from saved derived
outputs, never inside the model. Changing a unit cost, discount rate or analytic perspective
therefore never requires re-running MCMC — which matters when a reviewer asks for it months
after the analysis was run.

**Runs carry their provenance.** Every stored run records the git commit, whether the
working tree was dirty, the resolved configuration and a configuration hash. Posteriors may
only be reused by a model built from an identical configuration, and posterior subsampling
is seeded, so re-running an analysis reproduces its numbers exactly.

**Parameters live in spreadsheets, not code.** `parameters.xlsx` holds the value, prior,
source, LaTeX symbol and unit for each parameter, and the appendix parameter tables are
generated from the same file.

## Data

`data/shared/` contains UN World Population Prospects extracts. Country directories contain
study inputs; see each study's documentation for provenance. Economic inputs for Vietnam
were supplied by collaborators; enquiries about their reuse should be directed to the
corresponding author.

## Licence

BSD 2-Clause. See [LICENSE](LICENSE).
