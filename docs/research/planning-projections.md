---
title: Roadmap planning scenarios
status: planning
updated: 2026-10-25
---

# Roadmap planning scenarios

These charts support the [research roadmap](../NEXT.md). They are planning scenarios, not measured future results or forecasts of solve rates. Completed experimental evidence remains in the [E19 report](e19-funded-report.md).

![Left: a linear same-generator runtime scenario anchored at the completed E19-funded run. The shaded band is an assumed 20% sensitivity, not a statistical interval. Right: exact planned generator-forward budgets; neither quality nor wall-clock speed has been measured for early exit.](projections/compute-and-levels.svg)

## Training time at the current model size

The measured anchor is 54,419.81 training-loop seconds for 16.384M generator presentations on an A40. Holding generator size, batch, hardware and observed mean throughput constant gives:

| Generator presentations | Linear training scenario | Assumed ±20% sensitivity |
|---|---:|---:|
| 8.192M | 7.56 h | 6.05–9.07 h |
| 16.384M | 15.12 h | 12.09–18.14 h |
| 32.768M | 30.23 h | 24.19–36.28 h |
| 51.200M | 47.24 h | 37.79–56.69 h |

These are exposure scenarios for the current 6.44M generator. The last row matches the reported generator presentation count of **Original NCM (Kumbong et al.)**, not its 25.46M generator, tokenizer, recipe, training cost or expected accuracy. The scenarios exclude rental startup, transfer, data generation, encoding, codec training, final evaluation, failures and hardware changes. Checkpoint/validation overhead is included only through the observed mean training-loop time. The ±20% range is a chosen sensitivity assumption; a single completed run cannot establish a calibrated prediction interval. Longer schedules need their own explicit configuration and funding.

## Fixed levels and verified early exit

The roadmap proposes independent tests of 3, 6, 9 and 12 levels. One generator forward per level means exactly that many generator calls; the final codec decode is additional. This does **not** imply a proportional latency reduction: level lengths, prefix computation, attention, quantizer work and decoder overhead differ. A fixed level budget is distinct from stopping only after a Sudoku checker verifies a completed board.

Quality at these budgets is unknown. Measure solved boards, givens preservation, fallback frequency, level usage and synchronized end-to-end latency on the same inputs. Keep the full 12-level result as the baseline, retain per-board outcomes for paired comparisons, and preserve training/inference equivalence at every executed level. Do not infer early-exit quality from codec reconstruction of true solutions.

## Gates before larger claims

| Roadmap milestone | Required evidence before proceeding | Projection status |
|---|---|---|
| Seed replication | Fresh seed, same recipe and split; complete held-out evaluation | Current experiment; no quality forecast |
| Matched AR comparison | Matched generator presentations plus separate codec/system costs | Needs an explicit run allocation |
| Fixed levels and verified exit | Paired outcomes, checker validation, timing and full-depth fallback | Forward budgets shown above; quality unknown |
| Prefix caching and compilation | Exact output checks, both samplers measured under the same protocol | No supported speedup estimate yet |
| Real-text baseline | Dataset rights, tokenizer/padding contract, reproducible metrics and matched budgets | Sudoku timings and scores do not project to text |
| Upstream checkpoint reference | Independent format conversion and parity checks | No accuracy parity claim until measured |

The strongest completed small-generator finding is **98.10% versus 99.50% on easy boards**, using a **3.95× smaller generator** and **3.125× fewer generator presentations**. Medium and hard gaps remain 13.10 and 45.80 percentage points. More training might help, plateau or worsen held-out results; the current experiments do not identify a scaling law.

[Machine-readable assumptions](projections/scenarios.json) can be rebuilt with `uv run --group paper python scripts/project_roadmap.py`. Private account balances and spending plans are not included here.
