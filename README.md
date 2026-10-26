# Adaptive Interest-Weighted Diversification for Session-Based Recommendation under Dynamic User Intent

IRE course project (IIIT Hyderabad). We study whether **online estimation of a user's
changing category interests** (e.g. EWMA), used to weight item quality inside a
**Determinantal Point Process (DPP)** selector, improves the balance of relevance,
diversity, and responsiveness to intent drift. Experiments use a **controlled
simulation** where the true interest vector `z_t` is known. The recommender sees only
the interactions.

## Status
**Milestone 0 (research design): awaiting approval.** No code has been written yet.

| Document | Purpose |
|---|---|
| `IMPLEMENTATION_LOG.md` | Full, dated record of design, maths, implementation, tests, results (source for the report) |
| `DECISIONS.md` | Every important decision: alternatives, recommendation, approval status |
| `hypothesis.md` | Research hypotheses (created after they are agreed in Milestone 1) |

## Environment
Python 3.12, numpy, scipy, matplotlib, pandas. pytest for tests (to be installed,
pending approval). No GPU or deep learning frameworks needed.

## Commands
*None yet. They will be added as each component is implemented.*

## Prior work
An earlier exploratory GRU4Rec baseline was discarded when this repository was
started fresh. A summary is kept in `IMPLEMENTATION_LOG.md` §19.
