# Kiribati follow-up — strategic planning after PEARL

Source: ARC proposal, *"Population-wide screening offers a rare opportunity..."*. Read that
document before changing scenario definitions.

## Research question

Not "does screening work" but **what should happen next**. Once population-wide screening
has removed much of the prevalent disease and infection, the epidemiological landscape
changes: residual burden concentrates among people the intervention missed, and the relative
contributions of recent transmission, progression from existing infection, and delayed
diagnosis shift. The project asks how the optimal strategy changes *after* a population has
been screened, and how interventions should be **sequenced** as burden falls.

## Model

Identical to the published Kiribati model — including the reachability stratification — but
extended to a **20-year horizon** (the published analysis ran to 2035). The full Kiribati
output set is required here, unlike the other projects.

## Outcomes — two distinct classes

1. **Cumulative consequences** over the horizon: TB episodes and deaths averted.
2. **Epidemiological position reached at the horizon end**: residual reservoirs of disease
   *and* infection.

Keeping these separate is a stated requirement, not a presentational choice. Strategies with
similar cumulative impact may leave very different reservoirs, and therefore different
prospects for elimination.

## Strategies to compare

Business-as-usual services, against a cascade of enhanced and complementary strategies:

| Strategy | Model representation | Status |
| --- | --- | --- |
| Community awareness / clinical vigilance | Shorter diagnosis delay, i.e. raised passive detection rate | Shares the programmatic-lever mechanism with the funding-disruptions project, acting in the opposite direction |
| Household contact investigation + preventive treatment | **Not yet supported.** No household structure exists | Design decision outstanding |
| Repeat population screening | `ScreeningProgram` with varied timing, coverage, tool | Supported |
| Geographically focused screening in hotspots | **Not yet supported.** No spatial or risk-concentration dimension | Design decision outstanding |
| Dedicated outreach to people missed by PEARL | `ScreeningProgram` declaring its own `reachability` multipliers | Supported — see `stratifications/reachability.py` |
| Tongue swabs (Pluslife) as an alternative tool | `ScreeningTools.PLTS` | Exists but its sensitivities are placeholders flagged `skip` in the Kiribati parameter file |

Interventions must **not** be represented as idealised programmes with perfect
participation. Coverage, timing and implementation parameters are to be set with Kiribati
stakeholders, with plausible ranges evaluated to identify strategies robust to variation in
participation, diagnostic performance, treatment completion, and the characteristics of
populations reached or missed.

## Repeat infection testing — a specific complication

TST positivity persists after viable infection has cleared, and may be influenced by
previous testing and treatment. A second round of infection screening therefore cannot be
interpreted or implemented as the first was: some TST-positive people have already cleared
infection or completed preventive treatment, and re-treating them yields nothing.

The model already separates viable infection from TST positivity (`prev_se_cleared_tst`),
but it does **not** track who has previously been screened or treated. The proposal requires
testing alternative assumptions and decision rules for repeat testing. See PROGRESS.md for
the outstanding design decision.

## Adaptive sequencing

The goal is an adaptive intervention *pathway*: when broad screening should give way to
targeted approaches, when strengthened routine detection or contact investigation suffices,
and under what conditions renewed population-wide screening becomes necessary. Triggers may
be **time-based, epidemiological thresholds, or indicators of residual concentration**.

Time-based triggers are straightforward. Threshold-triggered switching is state-dependent
and is a known technical risk under JAX tracing — see PROGRESS.md.
