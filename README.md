# Adaptive Interest-Weighted Diversification for Session-Based Recommendation under Dynamic User Intent

IRE course project (IIIT Hyderabad). We study whether **online estimation of a user's
changing category interests** (e.g. EWMA), used to weight item quality inside a
**Determinantal Point Process (DPP)** selector, improves the balance of relevance,
diversity, and responsiveness to intent drift. Experiments use a **controlled
simulation** where the true interest vector `z_t` is known. The recommender sees only
the interactions.

## Status
**Milestone 5 (base relevance and similarity): complete, awaiting review.** The simulator, estimators, relevance scores and similarity kernel are implemented and tested. The proposed estimator is bias-corrected EWMA with alpha = 0.05 (D-27). The DPP and main experiments are not implemented yet.

| Document | Purpose |
|---|---|
| `IMPLEMENTATION_LOG.md` | Full, dated record of design, maths, implementation, tests, results (source for the report) |
| `DECISIONS.md` | Every important decision: alternatives, recommendation, approval status |
| `hypothesis.md` | Approved research hypotheses H1–H6 and the freeze rule |
| `literature/literature_matrix.md` | Verified literature matrix, gap analysis, reading list |
| `SIMULATION_DESIGN.md` | Simulator specification: catalogue, latent interests, drift scenarios, interaction model, seeds |

## Environment
Python 3.12, numpy, scipy, matplotlib, pandas. pytest for tests.
No GPU or deep learning frameworks needed.

## Commands
Run from this folder:
```bash
python -m pytest -q                  # run all tests
python experiments/sim_check.py      # simulator check on tuning seeds 0-4 -> results/m3_sim_check/<run_id>/
python experiments/estimator_study.py # estimator tuning/comparison on tuning seeds -> results/m4_estimators/<run_id>/
python experiments/relevance_check.py # Top-K scoring check on tuning seeds -> results/m5_relevance/<run_id>/
```

## Code layout
```
adaptdiv/      config, seeding, catalogue, scenarios, simulator   (Milestone 3)
               estimators, estimation_metrics                     (Milestone 4)
               relevance, similarity                              (Milestone 5)
experiments/   sim_check.py, estimator_study.py, ewma_init_comparison.py, relevance_check.py
tests/         pytest suite
results/       per-run outputs (config, summary, environment, plots)
```

## Prior work
An earlier exploratory GRU4Rec baseline was discarded when this repository was
started fresh. A summary is kept in `IMPLEMENTATION_LOG.md` §19.
