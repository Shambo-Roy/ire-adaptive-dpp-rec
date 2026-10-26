# IMPLEMENTATION_LOG.md

**Project:** Adaptive Interest-Weighted Diversification for Session-Based
Recommendation under Dynamic User Intent
**Course:** Information Retrieval and Extraction (IIIT Hyderabad)
**Author:** Shambo Roy. **Assistant:** Claude (AI collaborator; all decisions are
approved by the author.)

This is the source of truth for the final report. Entries are **appended and dated,
never deleted**. Superseded content is marked as superseded and kept.

Status tags used throughout:
- **[IMPL+TESTED]** implemented, with tests that were run and passed
- **[IMPL-UNTESTED]** implemented, not yet fully tested
- **[PLANNED]** designed or proposed, not implemented
- **[HYPOTHETICAL]** an expected or proposed result, *not* a measurement

---

## 1. Project objective and research question

**Objective.** Test whether (i) estimating a user's *changing* category-level
interests online, and (ii) using that estimate to weight item quality inside a
Determinantal Point Process (DPP) selector, improves the balance of relevance,
diversity, and responsiveness to intent change, compared with non-adaptive and
non-DPP baselines.

**Central research question (RQ).** Can adaptive interest estimation combined with
interest-weighted DPP selection improve the relevance–diversity–responsiveness
balance under changing user intent, compared with appropriate non-adaptive and
non-DPP baselines?

Proposed sub-questions (to be finalised with the hypotheses in M1):
- **RQ1 (adaptation):** Does an adaptive estimator (EWMA or sliding window) track `z_t`
  better than a non-adaptive one after drift, and at what cost under stable or noisy
  interests?
- **RQ2 (diversification):** Does DPP selection improve diversity, and how much
  relevance does it cost, for users with focused vs spread interests?
- **RQ3 (interaction):** Is the benefit of adaptation + DPP more than the sum of the
  separate effects (2×2 interaction)?

The project *investigates* the RQ. It does not assume the proposed method wins.

## 2. Current scope and approved design decisions

As of 2026-10-26 (Milestone 0): **no design decision has been approved yet.**
See `DECISIONS.md` (D-01 … D-18). Scope from the master prompt:
controlled simulation where the true `z_t` is known; open-loop evaluation
on fixed sequences; real-data validation is optional and needs approval.

## 3. Project architecture and data flow [PLANNED]

```
  config (JSON) ──► seeding (SeedSequence) ──┬─► catalogue: items, categories c_i, embeddings φ_i, popularity π_i
                                             └─► simulator: z_t trajectory (PRIVATE) ─► interactions x_1..x_T (OBSERVABLE)
                                                                       │                         │
                                       (evaluation only) ◄─────────────┘                         ▼
                                                                         estimator: ẑ_t = f(x_1..x_t)
                                                                                                 ▼
                                                     relevance r_i (from training users) ─► quality q_i = r_i · g(ẑ_{t,c_i})
                                                                                                 ▼
                                                                    selector: Top-K  |  greedy-MAP DPP on L = Diag(q) S Diag(q)
                                                                                                 ▼
                                      metrics(z_t, held-out x_{t+1..t+h}, list R_t) ─► results/<run_id>/
```

**Leakage firewall (design rule).** The simulator returns two separate objects:
`GroundTruth` (holds `z_t` and the user taste `u`) and `ObservedSequence` (item ids,
plus categories/features derivable from the catalogue). Estimators and recommenders
accept **only** `ObservedSequence` truncated at time `t`. Only the metrics module
receives `GroundTruth`. A test will check this.

## 4. Repository structure [PLANNED — awaiting D-05]

```
IRE Project/
├── README.md               setup, commands, overview
├── IMPLEMENTATION_LOG.md   this file
├── DECISIONS.md            decisions + approval status
├── hypothesis.md           (created after hypotheses are agreed, M1)
├── literature/             (M1) literature matrix
├── adaptdiv/               Python package
│   ├── config.py           dataclasses + JSON load/save
│   ├── seeding.py          SeedSequence stream management
│   ├── catalogue.py        item categories, embeddings, popularity
│   ├── scenarios.py        z_t trajectory generators (stable/gradual/abrupt/noisy/recurrent)
│   ├── simulator.py        interaction generation; GroundTruth vs ObservedSequence
│   ├── estimators.py       cumulative mean, EWMA, sliding window
│   ├── relevance.py        base relevance r_i, quality transform g
│   ├── similarity.py       S = ΦΦᵀ (+ jitter), PSD checks
│   ├── dpp.py              greedy MAP (incremental Cholesky) + naive reference
│   ├── recommenders.py     Top-K / DPP × static / adaptive / oracle
│   └── metrics.py          estimation, relevance, diversity, responsiveness
├── experiments/
│   ├── configs/*.json
│   └── run_experiment.py
├── analysis/               aggregation, statistics, plots
├── tests/                  pytest
└── results/<run_id>/       config.json, raw per-step metrics, summary, environment info
```

## 5. Mathematical definitions, equations, algorithms [PLANNED]

Notation: `C` categories, `N` items, item `i` has category `c_i`, embedding
`φ_i ∈ R^d` with `‖φ_i‖ = 1`, popularity `π_i`. User interest is
`z_t ∈ Δ^{C−1}` (the probability simplex). The observation at time `t` is item `i_t`,
with one-hot category vector `x_t = e_{c_{i_t}}`.

Candidate estimators (D-11):
- Cumulative mean: `ẑ_t = ẑ_{t−1} + (x_t − ẑ_{t−1}) / t`, which is the plain average
  of all observations.
- EWMA: `ẑ_t = (1−α) ẑ_{t−1} + α x_t`. Unrolled:
  `ẑ_t = (1−α)^t ẑ_0 + Σ_{s=1}^{t} α(1−α)^{t−s} x_s`, so an observation's weight
  decays geometrically with age, with a mean age of about `(1−α)/α`. It stays on the
  simplex if `ẑ_0` is on it.
- Sliding window: `ẑ_t = (1/min(t,W)) Σ_{s=max(1,t−W+1)}^{t} x_s`. EWMA and a window
  have the same mean age when `W ≈ 2/α − 1`.

Candidate DPP (D-12, D-14, D-15):
- `S = ΦΦᵀ` (cosine similarity, PSD). `L = Diag(q) S Diag(q)`.
- `det(L_Y) = (Π_{i∈Y} q_i²) · det(S_Y)`. The log-determinant splits into a quality
  term `Σ 2 log q_i` and a diversity term `log det S_Y`. `det S_Y` is the squared
  volume spanned by the selected embeddings, so near-duplicates shrink it.
- Greedy MAP: start from `Y = ∅` and, `K` times, add
  `j = argmax_{i∉Y} [log det L_{Y∪{i}} − log det L_Y]`. This is an approximation;
  exact MAP is NP-hard.
- Limits: `rank(S) ≤ d`, so at most `d` items can have a positive determinant. A
  category-indicator `S` has rank `C`.

## 6. Implementation details and important functions
*None yet.*

## 7. Data sources, preprocessing, data generation [PLANNED — D-03, D-09, D-10]
Recommended: a fully synthetic catalogue and interactions (see DECISIONS D-03, D-09,
D-10).

## 8. Experimental configurations and hyperparameters [PLANNED — D-18]
Preliminary: `C=10, N=500, d=32, K=10, T=200, U=100`; tuning seeds are kept separate
from evaluation seeds. Nothing has been fixed yet.

## 9. Baselines and ablation studies [PLANNED — D-16]
A 2×2 design of {static, adaptive} × {Top-K, DPP}, plus popularity, plus an oracle-z
diagnostic ceiling. Estimator variants: EWMA vs sliding window. Sensitivities: `α`/`W`,
noise `η`, interest concentration `κ`, drift speed `W`, and the DPP trade-off.

## 10. Evaluation metrics and calculation procedures [PLANNED — D-17]
Exact formulas will be written here at M7 and hand-checked in tests.

## 11. Commands used to run experiments
*None yet.*

## 12. Environment and dependency information
Recorded 2026-10-26 (Milestone 0 inspection):
- Windows 11 Pro 10.0.26200; 20 logical CPUs; 15.7 GB RAM; no GPU used.
- Python 3.12.10; numpy 2.5.2; scipy 1.18.0; pandas 3.0.5; matplotlib 3.11.1.
- torch 2.13.0+cpu is installed (from the earlier phase) and is **not needed** by the
  new design.
- **pytest is not installed** (proposed install, D-06).
- ~~Git repository; remote `origin = https://github.com/Shambo-Roy/ire-adaptive-session-rec.git`.~~
  (Superseded 2026-10-26: the author deleted that repository.)
- Git repository, freshly initialised 2026-10-26, branch `main`; remote
  `origin = https://github.com/Shambo-Roy/ire-adaptive-dpp-rec` (public).

## 13. Test cases and outcomes
*No tests yet.* Milestone 0 contains no executable code.

## 14. Experiment results
*No experiments run yet. No results exist.*

## 15. Problems encountered and resolutions
- **2026-10-26 (M0):** The working tree was found with every tracked file deleted
  (uncommitted deletions) and the git-ignored data, processed, and checkpoint folders
  missing. Nothing was restored or committed. The question was raised with the
  author as D-01.
- **2026-10-26 (M0), resolution of D-01:** the author deleted the old GitHub repo
  and chose a fresh start with no backup. The old local history was deleted and a
  new repository `ire-adaptive-dpp-rec` was created. The GRU4Rec work is permanently
  gone.
- **2026-10-26 (M0):** Possible dataset mislabelling in the earlier `DESIGN.md`
  ("RecSys Challenge 2015-style"). The file schema matches the CIKM Cup 2016
  DIGINETICA release. This is unverified and recorded in D-03.

## 16. Failed approaches and negative results
*None yet.*

## 17. Assumptions, limitations, threats to validity, reproducibility
Risks identified in Milestone 0 (mitigation in brackets):
1. **The simulator favours the method.** The recommender's category-weighted quality
   matches the category-driven generator. [Include a stable scenario where adaptation
   should *not* help, vary `κ` and `η`, add hidden item-level taste (`λ > 0`) that the
   recommender cannot model, and report every scenario.]
2. **Tuning on test.** [Use separate tuning and evaluation seeds. Tune every method
   with the same budget.]
3. **Leakage of `z_t` or future items.** [Use the API firewall (§3) plus a test.]
4. **Diversity measured by construction.** The DPP optimises the same `S` that ILD
   measures. [Treat ILD as descriptive; also report category coverage and
   relevance metrics.]
5. **Category alignment ≠ next-item accuracy.** [Report estimation metrics and
   next-item metrics separately.]
6. **Noise-floor bias.** The estimator tracks `(1−η) z + η/C`, not `z`. [Report it
   explicitly.]
7. **Approximate DPP.** Greedy MAP is not an exact optimum. [Describe it as greedy and
   test it against brute force on tiny `N`.]
8. **Kernel degeneracy** when `d < K` or with category-only features. [Require
   `d ≥ K`, add jitter, add tests.]
9. **High-variance single-target metrics.** [Use `h`-step targets, expected-utility
   metrics, many users and seeds, and paired tests.]
10. **External validity.** These are synthetic results only. [State this plainly. Real
    data can only be an optional, limited check.]

## 18. Decisions pending approval
D-02 … D-18 in `DECISIONS.md` (D-01 was decided 2026-10-26). **Blocking now:** D-02
(documentation), D-03 (data strategy), D-05 (repo structure), D-06 (pytest), and
D-07 (deadline). D-08 … D-18 are preliminary and will be confirmed at their
milestones.

## 19. Chronological milestone history

### Prior work (before this project definition) — superseded direction
- **2026-09-11, commit `8fe712f`:** "Phase 1: GRU4Rec baseline". A PyTorch GRU4Rec
  next-item model trained on a real session click-stream dataset (204,789 sessions
  after filtering, 43,136 items, 7-day/7-day time-based val/test split).
  Reported test results at that time: Recall@10 = 0.2622, Recall@20 = 0.3614,
  MRR@10 = 0.1093, MRR@20 = 0.1162. Full details are in `DESIGN.md` at that commit.
  The planned follow-ups then were drift injection by session splicing and
  hidden-state drift-response mechanisms (CUSUM, ADWIN, KL gate).
- That direction is **superseded** by the current master prompt (simulation,
  adaptive estimation, DPP). ~~The files are deleted from the working tree but kept in
  git history. Their handling is pending D-01.~~ **Update 2026-10-26:** by the
  author's decision (D-01) the code and history were permanently discarded. The
  numbers above can no longer be reproduced or verified, so they should be cited
  only as unverifiable prior exploration, if at all.

### Milestone 0 — Repository inspection and research design (2026-10-26)
- Inspected the repository, git history, and environment (findings in §12, §15, §19).
- Restated the RQ and sub-questions (§1).
- Identified 18 decisions and recorded alternatives and recommendations
  (`DECISIONS.md`).
- Proposed the architecture (§3), repo structure (§4), the maths to be implemented
  (§5), and validity risks (§17).
- Created `IMPLEMENTATION_LOG.md`, `DECISIONS.md`, and a new `README.md`. No code
  written, no tests run, no experiments run.
- **Status:** awaiting the author's review and approval.

## 20. Research questions ↔ hypotheses ↔ experiments ↔ conclusions
| RQ | Hypothesis | Experiment | Conclusion |
|---|---|---|---|
| RQ1 | *to be agreed in M1* | — | — |
| RQ2 | *to be agreed in M1* | — | — |
| RQ3 | *to be agreed in M1* | — | — |

---

## Appendix A — Viva notes by milestone

### Milestone 0
- **Why simulation?** Real logs don't reveal a user's true interest at each step, so
  estimation accuracy and adaptation lag can't be measured. Simulation gives a known
  `z_t` at the cost of external validity.
- **Why is the DPP not the estimator?** The DPP only selects a set given quality
  scores and similarities. Adaptivity enters only through `ẑ_t` in the quality
  scores.
- **Why must `S` be PSD?** A DPP is defined through principal minors of a PSD `L`.
  With `L = Diag(q) S Diag(q)`, `vᵀLv = (q∘v)ᵀ S (q∘v) ≥ 0` whenever `S` is PSD, so
  every determinant (and hence every probability) is non-negative.
- **Why the 2×2 design?** It separates the main effect of adaptation, the main
  effect of DPP, and their interaction, instead of comparing only "proposed vs
  baseline".
