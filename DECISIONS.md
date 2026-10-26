# DECISIONS.md — Design decisions, alternatives, rationale, approval status

Each decision has an ID (`D-xx`), the question, the alternatives, a recommendation,
the expected impact, and a **status**:

- `PENDING` — proposed, awaiting the student's approval. No code may assume it.
- `APPROVED` — approved by the student (date recorded).
- `APPROVED (modified)` — approved with changes (changes recorded).
- `REJECTED` — kept for the record with the reason.
- `PRELIMINARY` — a direction recorded early so later milestones are consistent;
  formal approval happens at the milestone named in the entry.

Decisions are never deleted. If one is revised, the old text stays and a dated
revision note is added below it.

Citations marked *(to verify, M1)* have not been checked against the source yet.
Do not use them in the report until Milestone 1 confirms them.

---

## Group A — Needed now (blocking Milestones 1–2)

### D-01 — What to do with the earlier GRU4Rec "Phase 1" work
**Status:** DECIDED 2026-10-26 by the author: fresh repository, old history discarded (see revision below)

**Context.** Commit `8fe712f` (2026-09-11, also on `origin/master`) contains a
GRU4Rec baseline trained on a real click-stream dataset (`config.py`, `data.py`,
`model.py`, `train.py`, `evaluate.py`, `DESIGN.md`, `README.md`,
`results/results.json`). In the current working tree **all of these files are
deleted (not committed)**, and the git-ignored folders (`archive/` raw data,
`processed/`, `checkpoints/`) are gone from disk too. The new master prompt
describes a different method (simulation + adaptive estimation + DPP, with no neural
network).

| Option | Pros | Cons |
|---|---|---|
| (a) Leave deleted; refer to commit `8fe712f` in the log | Clean slate; history is still in git/GitHub | The deletion has to be committed at some point; the old README gets replaced |
| (b) Restore into `legacy/phase1_gru4rec/` | Code stays visible; could later be a real-data baseline | Clutter; it's a PyTorch dependency the new design doesn't need |
| (c) Tag `8fe712f` as `phase1-gru4rec`, then (a) | Same as (a) plus an easy, named pointer for the report | One extra git command |

**Recommendation: (c).** The new research question doesn't need GRU4Rec. The tag
keeps the earlier work recoverable and citable as "prior exploratory work" in the
report. If real-data validation happens later (D-03), the old preprocessing in
`data.py` can be reused straight from the tag.
**Impact:** none on the new methodology. Note that the raw data files would have to
be supplied again for any real-data validation.

**Revision 2026-10-26 — DECIDED by the author (none of the options above):** the
author deleted the old GitHub repository (`Shambo-Roy/ire-adaptive-session-rec`) and
chose a **fresh start with no backup**. The local `.git` folder (the last copy of
commit `8fe712f`) was deleted and a new repository was initialised
(branch `main`, remote `Shambo-Roy/ire-adaptive-dpp-rec`, **public**).
**Consequence:** the GRU4Rec code, its notes, and its results no longer exist
anywhere. The figures quoted in `IMPLEMENTATION_LOG.md` §19 now come only from this
log and can't be re-checked. Any real-data validation (D-03) would need new
preprocessing code and the raw files supplied again.

### D-02 — Documentation: `IMPLEMENTATION_LOG.md` vs the older `DESIGN.md`
**Status:** APPROVED 2026-10-26 (as recommended)

The earlier phase used a living `DESIGN.md`, with a standing rule never to delete
entries for superseded components. The new prompt makes `IMPLEMENTATION_LOG.md` the
central record.
**Recommendation:** make `IMPLEMENTATION_LOG.md` the single source of truth and keep
the same never-delete rule. Its "Prior work" section summarises `DESIGN.md` and
points to commit `8fe712f` instead of keeping two parallel logs.

### D-03 — Catalogue and data strategy
**Status:** APPROVED 2026-10-26 (as recommended: option (a) synthetic for main experiments; (c) optional at M9, needs separate approval)

| Option | Pros | Cons |
|---|---|---|
| (a) **Synthetic catalogue + synthetic interactions** | Full control; ground-truth `z_t` known; item features designed to suit DPP; no licensing issues; fast | External validity limited to the model's assumptions |
| (b) Real catalogue (e.g. MovieLens genres, DIGINETICA categories) + synthetic interactions | Realistic category sizes and popularity skew | Real catalogues rarely have usable content embeddings, so features would still be synthetic; MovieLens genres are multi-label (no single `c_i`); licence and download steps |
| (c) Real session dataset for validation | Some real-world evidence | No ground-truth `z_t`; only next-item and diversity metrics; preprocessing effort |

**Recommendation: (a) for every main experiment.** (c) is an optional Milestone 9
check, only if time allows, and needs a separate approval. Whatever real
dataset is chosen must have its licence and fields verified first. Note:
the earlier phase's files (`train-item-views.csv` + `product-categories.csv`,
schema `sessionId;userId;itemId;timeframe;eventdate`) look like the
**CIKM Cup 2016 / DIGINETICA** dataset, even though the old `DESIGN.md` called them
"RecSys Challenge 2015-style". The identity and licence must be verified before any
use.

### D-04 — Evaluation protocol: open-loop, fixed sequences
**Status:** PRELIMINARY (given by the master prompt; recorded for completeness)

Sequences are generated once per (scenario, seed, user). Every method is evaluated
on exactly the same sequences, and recommendations are never fed back into the
simulator. A closed-loop (interactive) simulation is an optional extension that
needs approval.

### D-05 — Repository structure
**Status:** APPROVED 2026-10-26 (as recommended). See `IMPLEMENTATION_LOG.md` §4 for the tree.
**Recommendation:** a small flat Python package (`adaptdiv/`), JSON experiment
configs (to avoid a PyYAML dependency), `tests/` with pytest, and `results/<run_id>/`
folders that each hold the exact config used.

### D-06 — Dependencies
**Status:** APPROVED 2026-10-26 (as recommended; pytest installed)

Already installed: Python 3.12.10, numpy 2.5.2, scipy 1.18.0, pandas 3.0.5,
matplotlib 3.11.1. **pytest is not installed.**
**Recommendation:** use numpy, scipy (statistical tests), matplotlib, and pandas
(result tables only). Install **pytest** as the one new dependency (dev only). No
PyTorch or neural networks.

### D-07 — Deadline and compute budget
**Status:** PENDING. **Needs input from the student:** the submission deadline and the
viva date. The experiment scale (D-17) and whether Milestone 9 happens depend on
these. Machine: 20 logical CPUs, 15.7 GB RAM, no GPU.

---

## Group B — Preliminary directions (formal approval at the milestone named)

### D-08 — Representation of the latent interest `z_t` (approve at M2)
**Recommendation:** a probability vector on the simplex over `C` categories
(`z_t ≥ 0`, `Σ_c z_{t,c} = 1`). Alternative: an unnormalised intensity vector,
which is harder to interpret, so it isn't recommended.

### D-09 — Interaction-generation model (approve at M2)
Preliminary proposal (all parameters go in config):
1. With probability `1−η`, draw a category `c_t ~ Categorical(z_t)`; with probability
   `η` (noise), draw `c_t ~ Uniform(C)`.
2. Within that category, draw an item with
   `P(i | c_t) ∝ π_i^γ · exp(λ ⟨φ_i, u⟩)`, where `π_i` is item popularity, `φ_i` the
   item embedding, and `u` a hidden per-user taste vector. `λ = 0` gives a pure
   popularity-within-category model. `λ > 0` adds item-level taste that the
   recommender cannot see directly, which is **deliberate partial misspecification**
   so that the recommender's model is not identical to the simulator.
**Alternatives:** an embedding-only generator (no categories); a Dirichlet–multinomial
model with a random-walk `z_t`.

### D-10 — Drift scenarios (approve at M2)
Stable (fixed `z`), gradual (linear interpolation `z_A → z_B` over `W` steps),
abrupt (switch at `τ`), noisy (stable or drifting `z` with higher `η`), and recurrent
(`A → B → A`). Each user's `z_A`, `z_B` are sampled from a Dirichlet distribution
with concentration `κ`. **`κ` (focused vs spread interests) is a sensitivity
factor**, because diversification should help most for spread interests and least
for focused ones.

### D-11 — Interest estimator (approve at M4)
Candidates, all using `x_t` = the one-hot category of the observed item:
- **Cumulative mean** (non-adaptive): `ẑ_t = ẑ_{t−1} + (x_t − ẑ_{t−1})/t`
- **EWMA** (proposed adaptive): `ẑ_t = (1−α) ẑ_{t−1} + α x_t`
- **Sliding window** (alternative adaptive): category frequencies over the last `W`
  interactions
**Recommendation:** implement all three. With `ẑ_0` uniform and one-hot `x_t`, EWMA
stays on the simplex automatically (a convex combination), so it never needs
renormalisation. Compare EWMA and the window at matched memory (`W ≈ 2/α − 1`).
Tune `α` and `W` on tuning seeds only. A change-point-based adaptive `α` is an
optional extension.

### D-12 — Item representation and similarity `S` (approve at M5)
**Recommendation:** each item has one category `c_i` and an embedding
`φ_i = normalise(μ_{c_i} + σ ε_i)` in `R^d`. Set `S = Φ Φᵀ`, i.e. cosine similarity.
This is a Gram matrix, so it is PSD by construction.
**Important:** a category-indicator kernel `S_ij = 1[c_i = c_j]` is PSD but has
rank `C`, so a DPP could select at most one item per category, and `K > C` would be
impossible. Likewise `rank(ΦΦᵀ) ≤ d`, so we need **`d ≥ K`** (proposal: `d = 32`,
`K = 10`) and a small jitter `εI` for numerical stability.

### D-13 — Base relevance `r_i` and candidate set (approve at M5)
| Option | Pros | Cons |
|---|---|---|
| (a) Popularity estimated from independent "training" simulated users | Observable, non-personalised, keeps the 2×2 design clean | Weak on its own |
| (b) Content similarity between `φ_i` and the mean embedding of recent history | Personalised | It is itself an adaptive profile, which blurs the "adaptive vs static" factor |
| (c) Item–item co-occurrence (item-kNN) from training sessions | A standard session-rec baseline | More code; also history-dependent |
**Recommendation: (a)**, with `r_i > 0` (smoothed counts). Candidate set: all items
(N ≈ 500), excluding items already consumed in the current sequence (to be confirmed
at M5).

### D-14 — Quality transform and DPP kernel (approve at M6)
`L = Diag(q) S Diag(q)`, i.e. `L_ij = q_i q_j S_ij`. `L` is PSD whenever `S` is PSD
and `q` is real. Preliminary choice: `q_i = r_i · (ẑ_{t,c_i} + ε)^β`, with `ε > 0`
so that no item gets zero quality just because its category hasn't been seen yet.
A trade-off parameter between relevance and diversity, of the kind used by Chen et
al. (2018) *(metadata verified, lit. A5; the exact trade-off form must be checked in the full text before M6)*, will be specified and tuned the same way for every
DPP variant.

### D-15 — DPP inference algorithm (approve at M6)
**Recommendation:** greedy MAP inference with incremental Cholesky updates (Chen,
Zhang & Zhou, NeurIPS 2018 *(metadata verified, lit. A5)*). Cost (to confirm from the full text) is about `O(N K²)` after building the
kernel. It must be described as an **approximate (greedy) MAP**, not an exact one:
exact MAP for DPPs is NP-hard (Ko, Lee & Queyranne 1995, lit. A3; NP-hardness of max-det subset selection). Alternatives: exact
k-DPP sampling (stochastic, so it adds variance to evaluation) and naive greedy with
repeated determinants (`O(N K⁴)`; useful only as a test oracle).

### D-16 — Methods compared (approve at M6)
A 2×2 factorial design plus two reference points:

| | Top-K | DPP |
|---|---|---|
| **Static profile** (cumulative mean) | Static Top-K | Static DPP |
| **Adaptive profile** (EWMA; window as a variant) | Adaptive Top-K | **Adaptive DPP (proposed)** |

- **Popularity Top-K** (no profile): the floor.
- **Oracle-z Top-K / Oracle-z DPP**: uses the true `z_t`. **This is a diagnostic
  ceiling only, not a competing method.** It separates estimation error from
  selection effects. It is clearly labelled and never fed to any real method.
- Optional: MMR as a cheaper diversification baseline, to test whether the DPP
  specifically is what helps.

### D-17 — Metrics and targets (approve at M7)
- **Estimation:** L1 or total-variation distance and RMSE between `z_t` and `ẑ_t`.
  Note that the estimator tracks the *observed* category distribution
  `(1−η) z_t + η/C`, so a noise-floor bias exists even for a perfect tracker. This
  will be reported, not hidden.
- **Next-item relevance:** Recall@K and NDCG@K against the next `h` held-out
  interactions (proposal: `h = 5`).
- **Expected utility (simulator-side, evaluation only):** `Σ_{i∈R} P(i | z_{t+1})`,
  which is less noisy than a single next item.
- **Diversity:** intra-list diversity `1 − cos(φ_i, φ_j)` and category coverage.
  Risk: the DPP directly optimises the same similarity that ILD measures, so ILD gains
  are partly true by construction. Report them, but don't treat them as the success
  criterion.
- **Responsiveness:** adaptation lag, i.e. the number of interactions after a shift
  until estimation error stays below a threshold `δ` (defined relative to the
  stable-state error) for `m` consecutive steps. Also post-shift cumulative
  Recall/NDCG.
- **Statistics:** paired comparisons across seeds (Wilcoxon signed-rank or a paired
  bootstrap CI), Holm correction, and effect sizes.

### D-18 — Scale and seed protocol (approve at M3 and M8)
Preliminary: `C = 10`, `N = 500`, `d = 32`, `K = 10`, `T = 200` interactions per
user, `U = 100` users per scenario. **Tuning seeds** are kept separate from
**evaluation seeds** (e.g. 0–4 vs 100–119). `numpy.random.SeedSequence` spawns
independent streams for the catalogue, trajectories, and interactions.
Rough cost estimate (to be measured, not assumed): 5 scenarios × 100 users × 200
steps × 20 seeds × ~8 methods ≈ 16M recommendation calls. With
N = 500 and K = 10 greedy DPP, this is likely tens of minutes to a few CPU-hours.
Evaluating every `s`-th step, or running in parallel over seeds, brings it down.
This will be benchmarked before asking to run the main experiments.

---

## Group C — Raised in Milestone 1 (literature review)

### D-19 — Add a calibration re-ranker (Steck 2018) as a baseline?
**Status:** PENDING (raised 2026-10-26, M1)

**Why it came up.** Steck (2018, literature B7) re-ranks a list so that its category
distribution matches the user's interest distribution. Lin et al. (2024, B8) already
contrast static and time-windowed profiles for calibration. Without this baseline, a
reviewer can ask: "If you just want the list to follow `ẑ_t`, why not calibrate? Is
the DPP needed?"

| Option | Pros | Cons |
|---|---|---|
| (a) No calibration baseline | Less work | Leaves the "why a DPP?" question open |
| (b) **Adaptive calibration re-ranker** (greedy, KL to `ẑ_t`), the same estimator fed to a different selector | Directly tests whether the DPP adds anything beyond matching category proportions | About one extra day of work. Steck's paper must be read in full first |
| (c) Calibration and MMR both | Most complete | More methods and more multiple-comparison burden |

**Recommendation: (b)**, run as a secondary comparison (not part of the 2×2). MMR stays
optional.

### D-20 — Hypotheses H1–H6
**Status:** PENDING. Full text is in `IMPLEMENTATION_LOG.md` §20. `hypothesis.md`
will be created only after approval. Exact metric formulas are fixed at M7 and the
hypotheses are frozen (pre-registered) before any evaluation-seed run at M8.

### D-21 — Statistical protocol and practical significance
**Status:** PENDING

- **Unit of analysis:** seed-level means (users averaged within a seed), paired
  across methods because every method sees the same sequences. Alternative: user-level
  analysis with seed as a cluster (more power, more complex).
- **Test:** two-sided paired Wilcoxon signed-rank plus a paired bootstrap 95% CI of
  the difference. Holm correction across the confirmatory family H1–H6.
- **Smallest effect size of interest (SESOI), proposed:** relative difference
  ≥ 3% in NDCG@K or expected utility; ≥ 2 interactions in adaptation lag;
  ≥ 0.02 absolute in mean total-variation error. A result counts as *practically
  meaningful* only if the whole CI lies beyond the SESOI.
- The SESOI values may be revised at M7 using **tuning seeds only**, never after
  seeing evaluation-seed results.

---

## Revision history
- 2026-10-26 — File created in Milestone 0. All entries are PENDING or PRELIMINARY.
- 2026-10-26 — D-01 decided (fresh repo). D-02, D-03, D-05 and D-06 approved.
- 2026-10-26 — Milestone 1: added D-19 (calibration baseline), D-20 (hypotheses) and
  D-21 (statistical protocol). Citations in D-14/D-15 checked at metadata level (see
  `literature/literature_matrix.md`, A3 and A5). The trade-off parametrisation in D-14
  is still unverified until Chen et al. is read in full.
