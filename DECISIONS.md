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
**Status:** APPROVED 2026-10-26 (as proposed)
**Recommendation:** a probability vector on the simplex over `C` categories
(`z_t ≥ 0`, `Σ_c z_{t,c} = 1`). Alternative: an unnormalised intensity vector,
which is harder to interpret, so it isn't recommended.
**M2 update (2026-10-26):** proposed for approval unchanged. See `SIMULATION_DESIGN.md` §3.

### D-09 — Interaction-generation model (approve at M2)
**Status:** APPROVED 2026-10-26 (as proposed)
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
**M2 update (2026-10-26), proposed for approval with two refinements**
(`SIMULATION_DESIGN.md` §5):
(i) the taste term uses the *standardised* score `ŝ_i(u)`, because the raw `⟨φ_i,u⟩` is
tiny in `d = 32` (sd ≈ 1/√d) and would make `λ` hard to interpret;
(ii) defaults `γ = 1`, `λ = 0.5` (sensitivity `λ ∈ {0, 1}`), base noise `η = 0.05`,
with noise categories drawn uniformly. Alternative noise model, not recommended:
noise categories drawn by global popularity ("trending" clicks), which is more
realistic but confounds noise with popularity.

### D-10 — Drift scenarios (approve at M2)
**Status:** APPROVED 2026-10-26 (as proposed)
Stable (fixed `z`), gradual (linear interpolation `z_A → z_B` over `W` steps),
abrupt (switch at `τ`), noisy (stable or drifting `z` with higher `η`), and recurrent
(`A → B → A`). Each user's `z_A`, `z_B` are sampled from a Dirichlet distribution
with concentration `κ`. **`κ` (focused vs spread interests) is a sensitivity
factor**, because diversification should help most for spread interests and least
for focused ones.
**M2 update (2026-10-26), proposed for approval** (`SIMULATION_DESIGN.md` §4):
`T = 200`; abrupt `τ = 100`; gradual ramp over `t = 80…120` (`W = 40`); recurrent
`A → B → A` with switches at 70 and 140; noisy = stable `z_A` with `η = 0.3`. Noise
acts on interactions, never on `z_t`. Concentration is a crossed factor
`κ ∈ {0.1, 1.0}` (D-23), and `z_B` is a half-swap permutation of `z_A` (D-23).

### D-11 — Interest estimator (approve at M4)
**Status:** APPROVED 2026-10-28 (as recommended). Implemented in M4 (`adaptdiv/estimators.py`). See D-27 for an issue found during M4.
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
**Status:** APPROVED 2026-11-01 (M5 proposal below)
**Recommendation:** each item has one category `c_i` and an embedding
`φ_i = normalise(μ_{c_i} + σ ε_i)` in `R^d`. Set `S = Φ Φᵀ`, i.e. cosine similarity.
This is a Gram matrix, so it is PSD by construction.
**Important:** a category-indicator kernel `S_ij = 1[c_i = c_j]` is PSD but has
rank `C`, so a DPP could select at most one item per category, and `K > C` would be
impossible. Likewise `rank(ΦΦᵀ) ≤ d`, so we need **`d ≥ K`** (proposal: `d = 32`,
`K = 10`) and a small jitter `εI` for numerical stability.
**M5 proposal (2026-11-01), PENDING approval:** unchanged: `S = ΦΦᵀ + εI` with
`ε = 1e-6` on the public embeddings, `d = 32`, `K = 10`. Tests: symmetry, unit diagonal
(before jitter), positive semidefinite, rank ≤ `d`.

### D-13 — Base relevance `r_i` and candidate set (approve at M5)
**Status:** APPROVED 2026-11-01 (M5 proposal below: all items as candidates, within-category popularity; global popularity only for the Popularity Top-K floor)
| Option | Pros | Cons |
|---|---|---|
| (a) Popularity estimated from independent "training" simulated users | Observable, non-personalised, keeps the 2×2 design clean | Weak on its own |
| (b) Content similarity between `φ_i` and the mean embedding of recent history | Personalised | It is itself an adaptive profile, which blurs the "adaptive vs static" factor |
| (c) Item–item co-occurrence (item-kNN) from training sessions | A standard session-rec baseline | More code; also history-dependent |
**Recommendation: (a)**, with `r_i > 0` (smoothed counts). Candidate set: all items
(N ≈ 500), excluding items already consumed in the current sequence (to be confirmed
at M5).
**Superseded note (2026-11-01):** the exclusion above conflicts with approved D-24, which
says recommenders do **not** exclude consumed items (excluding them while repeats are
generated would make those targets unreachable). D-24 governs unless changed at M5.
**M5 proposal (2026-11-01), PENDING approval:**
- **Candidates:** all `N` items, for every method. No exclusion (D-24).
- **Base relevance:** popularity from the training population, **within category**:
  `r_i = (count_i + 1) / (Σ_{j∈c_i} count_j + n_c)`. Then `r_i · ẑ_{t,c_i}` is a plug-in
  estimate of the next-item probability, and only `ẑ_t` carries category preference.
  The D-26 global `r_i` is kept only for the **Popularity Top-K** floor, which has no
  profile.
- **Why not global `r_i` everywhere:** a check on tuning seeds (2026-11-01) found the
  category totals of global `r_i` range about 0.09–0.11 (max/min ≈ 1.2) although categories
  are symmetric by design. That is training-sample noise, and it would act as a fake
  category preference. Within-category estimates correlate 0.996 with true `π_i` (mean
  within-category TV 0.037).
- **Consequence to accept:** `r_i` is close to true popularity, so the ranking model is
  close to the simulator except for the hidden taste `λ` (risk 1 in the log).
- **Repeat-rate risk (risk 15):** report the Popularity Top-K floor, and fix any no-repeat
  metric variant at M7.
Alternatives: global `r_i` everywhere (simpler, keeps the noise above); content similarity
or item-kNN (rejected in D-13 because they blur the static vs adaptive factor).

### D-14 — Quality transform and DPP kernel (approve at M6)
**Status:** APPROVED 2026-11-01 (M6 proposal below)
`L = Diag(q) S Diag(q)`, i.e. `L_ij = q_i q_j S_ij`. `L` is PSD whenever `S` is PSD
and `q` is real. Preliminary choice: `q_i = r_i · (ẑ_{t,c_i} + ε)^β`, with `ε > 0`
so that no item gets zero quality just because its category hasn't been seen yet.
A trade-off parameter between relevance and diversity, of the kind used by Chen et
al. (2018) *(metadata verified, lit. A5; the exact trade-off form must be checked in the full text before M6)*, will be specified and tuned the same way for every
DPP variant.

**M6 proposal (2026-11-01), PENDING approval.** Full-text check of Chen et al.: their
trade-off is **linear** in the score, `θ Σ r_i + (1−θ) log det S_Y`. Our scores are small
probabilities, so that scale doesn't fit. Proposal: the **log-score** form

`F(Y) = θ Σ_{i∈Y} log s_i + (1−θ) log det S_Y`,  `s_i = r_i (ẑ_{t,c_i} + ε)`,  `ε = 0.01`, `θ ∈ (0, 1]`.

- **It is a valid DPP** for `θ < 1`: `L = Diag(q) S Diag(q)` with
  `q_i = s_i^{θ/(2(1−θ))} > 0`, because `log det L_Y = (θ/(1−θ)) Σ log s_i + log det S_Y`.
  This is Kulesza & Taskar's log-linear quality with feature `log s_i`. `S` is PSD and `q`
  is positive, so `L` is PSD.
- **`θ = 1` is exactly Top-K**, so Top-K and DPP sit on one continuous scale.
- `ε` keeps quality positive when `ẑ_{t,c} = 0`. `S` is unchanged (D-12); Kulesza &
  Taskar allow signed similarity, so Chen's mapping into `[0,1]` isn't needed.

| Option | Pros | Cons |
|---|---|---|
| (a) Linear (Chen): `θ Σ s_i` | Exactly as published | Prototype: `θ` 1.0 → 0.95 drops focused-stable coverage 0.369 → 0.287, and below 0.9 lists ignore relevance (cats 7.9–9.3). Would need score rescaling, one more design choice |
| (b) No trade-off: `q_i = s_i` | No parameter | Fixes the balance at `θ = 2/3`; can't trace a trade-off curve |
| (c) **Log-score form** | Smooth, scale-free, has Top-K as a limit | Differs from the published form, so it must be stated as our adaptation |

Prototype evidence (throwaway, tuning seed 1, 20 users, every 10th step, `K = 10`; lists
for abrupt are steps 100–140): log form, `θ` = 1.0 / 0.7 / 0.5 / 0.3: focused stable
coverage 0.369 / 0.368 / 0.363 / 0.335, categories 1.9 / 2.0 / 2.2 / 2.6; spread stable
coverage 0.170 / 0.167 / 0.162 / 0.148, categories 3.4 / 3.9 / 4.6 / 5.6. Spread abrupt
coverage stayed 0.132 / 0.133 / 0.131 / 0.125 while categories rose 4.0 / 4.7 / 5.3 / 6.6.
Small sample: design evidence only.
**Recommendation: (c).** `θ` is shared by all DPP variants, swept over
{0.3, 0.5, 0.6, 0.7, 0.8, 0.9} at M6 and selected on tuning seeds once M7 fixes the metrics.

### D-15 — DPP inference algorithm (approve at M6)
**Status:** APPROVED 2026-11-01 (M6 proposal below)
**Recommendation:** greedy MAP inference with incremental Cholesky updates (Chen,
Zhang & Zhou, NeurIPS 2018 *(metadata verified, lit. A5)*). Cost (to confirm from the full text) is about `O(N K²)` after building the
kernel. It must be described as an **approximate (greedy) MAP**, not an exact one:
exact MAP for DPPs is NP-hard (Ko, Lee & Queyranne 1995, lit. A3; NP-hardness of max-det subset selection). Alternatives: exact
k-DPP sampling (stochastic, so it adds variance to evaluation) and naive greedy with
repeated determinants (`O(N K⁴)`; useful only as a test oracle).

**M6 proposal (2026-11-01), PENDING approval.** Chen et al. Algorithm 1 (maintain `c_i`,
`d_i²`), run on `S` with gain `θ log s_i + (1−θ) log d_i²`, as Chen et al. describe for their
modified objective. This avoids overflow from `q` and also handles `θ = 1`. Fixed list
size `K`, deterministic. Floor `d_i²` at 1e-12 before the log. Confirmed from the full
text: `O(K²N)` to return `K` of `N` items. "Exact" in that paper means exact *greedy*.
Report wording: **greedy approximate MAP**: MAP is NP-hard to approximate within `8/9+ε`,
and greedy's known ratio is only `O(1/k!)` (Kulesza & Taskar Thm 2.9 and §2.4.5).
Tests: identical picks to naive determinant greedy, and comparison with brute force on
small `N`. Prototype speed: about 0.2–0.4 ms per list at `N = 500`, `K = 10`.
Rejected: k-DPP sampling (adds randomness to evaluation; needs an eigendecomposition).
**Recommendation:** as above.

### D-16 — Methods compared (approve at M6)
**Status:** APPROVED 2026-11-01 (M6 proposal below)
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

**M6 proposal (2026-11-01), PENDING approval:**
| Role | Methods |
|---|---|
| Confirmatory 2×2 | Static Top-K, Adaptive Top-K, Static DPP, Adaptive DPP (static = cumulative mean; adaptive = bias-corrected EWMA) |
| Floor | Popularity Top-K (global popularity, no profile) |
| Diagnostic ceiling | Oracle-z Top-K and Oracle-z DPP (true `z_{t+1}`; never a competitor) |
| Secondary | Adaptive calibration re-ranker (D-19; **blocked until Steck's paper is available**) |
| Ablations | Uniform-start EWMA; window(40) |
| Dropped | MMR (fewer comparisons; can return at M9) |

### D-17 — Metrics and targets (approve at M7)
**Status:** APPROVED 2026-11-01 (M7 proposal below; it replaces the preliminary text where they differ)
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
**M8 scale APPROVED 2026-11-02 by the author; run as proposed below (812 s).**
**M8 proposal (2026-11-01), PENDING approval:** `U = 100` per cell, 20 evaluation seeds
(100–119), every step, default config, all D-16 methods except calibration. Measured basis:
the θ run did 15 methods on 5 seeds in 392 s using 5 processes. M8 has 5 DPP methods on 20 seeds,
estimated at about 15 minutes with 5 processes.

**Status:** Seed protocol APPROVED 2026-10-26 (as refined in `SIMULATION_DESIGN.md` §7: tuning seeds 0–4, evaluation seeds 100–119, SeedSequence streams, common random numbers across scenarios). Final scale (U, number of seeds) is still to be confirmed at M8 after benchmarking.
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
**Status:** APPROVED 2026-10-26 (option (b): adaptive calibration re-ranker as a secondary comparison; Steck 2018 must be read in full before implementing)

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
**Status:** APPROVED 2026-10-26. Recorded in `hypothesis.md` (frozen before any evaluation-seed run). Full text is also in `IMPLEMENTATION_LOG.md` §20.
Exact metric formulas are fixed at M7 and the hypotheses are frozen (pre-registered)
before any evaluation-seed run at M8.

### D-21 — Statistical protocol and practical significance
**Status:** APPROVED 2026-10-26 (as proposed; SESOI may be revised at M7 using tuning seeds only)

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

## Group D — Raised in Milestone 2 (simulation design)

All APPROVED 2026-10-26 as proposed (raised the same day). The full specification is in
`SIMULATION_DESIGN.md`. The seed protocol there refines D-18 (common random numbers
across scenarios, stream layout), and that refinement is also pending.

### D-22 — Structure of category centroids
**Status:** APPROVED 2026-10-26 (as proposed)
| Option | Pros | Cons |
|---|---|---|
| (a) **Independent random centroids** (nearly orthogonal in `d = 32`) | Simple. "Different category" means "dissimilar", so ILD and category coverage agree | No related-category structure |
| (b) Grouped centroids (e.g. 5 groups × 2 sibling categories with correlation `ρ`; MC: `ρ = 0.5`, `σ = 1` gives sibling cosine 0.12 vs 0.00 for unrelated) | Makes embedding diversity differ from category coverage, a more realistic test of the DPP | One more parameter and harder to explain |
**Recommendation:** (a) for the main experiments, (b) as an optional sensitivity
check in M9.

### D-23 — Interest concentration levels and shift construction
**Status:** APPROVED 2026-10-26 (as proposed)
- **Concentration:** `κ ∈ {0.1 (focused), 1.0 (spread)}` as a crossed factor in every
  scenario. MC check: on average 2.0 vs 5.2 categories hold 80% of the mass.
  `κ = 0.3` is an optional middle level.
- **Shift:** `z_B` = half-swap permutation of `z_A`. It keeps concentration unchanged
  and gives large shifts for both levels (mean TV 0.99 / 0.65). The alternative,
  swapping only the top 2 categories, gives mean TV 0.36 for spread users, and
  sometimes as little as 0.16, which is too weak to test adaptation.
  A drawn-independently `z_B ~ Dirichlet` would confound the shift with a change in
  concentration.
**Recommendation:** as above.

### D-24 — Repeat consumption
**Status:** APPROVED 2026-10-26 (as proposed)
| Option | Pros | Cons |
|---|---|---|
| (a) **Repeats allowed** in the simulator; recommenders do not exclude already-consumed items | Consistent (a target can always be hit); no category exhaustion | Lists may re-recommend items already seen. Less realistic for some domains |
| (b) No repeats, and recommenders exclude consumed items | Realistic for e-commerce "discovery" | MC check: focused users would exhaust a 50-item category (140–180 draws expected); it would need `N` of about 2,000+ and distorts `P(i \| c)` over time |
| (c) Repeats allowed, but recommenders exclude consumed items | — | **Inconsistent**: targets that are repeats could never be hit, which biases Recall |
**Recommendation: (a)**, stated as a limitation.

### D-25 — Hyperparameter tuning without leaking the scenario
**Status:** APPROVED 2026-10-26 (as proposed)
**Recommendation:** tune one setting per method (`α`, `W`, DPP trade-off, etc.) on a
**pooled mix of all scenarios and both `κ` levels** using tuning seeds 0–4.
Per-scenario tuned settings are reported only as an "oracle-tuned" sensitivity
result, because choosing `α` per scenario uses the scenario label, which is
prohibited information.

### D-26 — Training population for base relevance `r_i`
**Status:** APPROVED 2026-10-26 (as proposed)
**Recommendation:** for each seed, `U_train = 1,000` users × 50 interactions, with
stable interests drawn from the same Dirichlet mixture (both `κ` levels), the same
catalogue, and separate random streams from the evaluation users. Then
`r_i = (count_i + 1)/(Σ counts + N)`.
This fixes how `r_i` is *produced*. Whether `r_i` = popularity is the final relevance
model remains D-13, approved at M5.

---

## Group E — Raised in Milestone 4 (estimators)

### D-27 — EWMA initialisation: uniform start vs bias-corrected start
**Status:** APPROVED 2026-11-01: option (b), bias-corrected EWMA with `α* = 0.05` is the proposed adaptive estimator; the uniform-start EWMA (`α = 0.07`) stays as an ablation. This refines D-11. Raised 2026-10-28, M4.

**What we found** (M4 study, tuning seeds; `IMPLEMENTATION_LOG.md` §14). Approved D-11 starts
EWMA from `ẑ_0 = uniform`. That start fades as `(1−α)^t`: about 23% of it remains
after 20 steps at `α = 0.07`. The cumulative mean and the window use only the
observations from `t = 1`. Consequences:
- Focused users (`κ = 0.1`): EWMA's early error is much higher (the plot starts at
  0.67 vs 0.50). Its stable-scenario error after burn-in is 0.121 vs 0.076 for the
  cumulative mean. Part of that gap is initialisation, not forgetting.
- Spread users (`κ = 1.0`): the uniform start *helps*, because spread interests are
  close to uniform.
- Small `α` is penalised most (`α = 0.02` has a stable error of 0.173 for focused
  users, worse than `α = 0.05`). This can distort `α` tuning and the H2 comparison
  ("cost of adaptation"): H2 would partly measure the prior, not forgetting.

| Option | Definition | Pros | Cons |
|---|---|---|---|
| (a) Keep the uniform start (as approved) | `ẑ_0 = 1/C`, then the EWMA update | Already implemented; simple; never produces exact zeros early | Confounds H2 with initialisation; the comparison with the cumulative mean is unequal |
| (b) **Bias-corrected EWMA** | `m_t = (1−α) m_{t−1} + α x_t`, `m_0 = 0`; `ẑ_t = m_t / (1 − (1−α)^t)`; `ẑ_0` = uniform | Uses observations only, like the other two estimators. Still a probability vector. It equals the normalised exponentially weighted average of `x_1..x_t`. With `α → 0` it becomes the cumulative mean | One more line of maths to explain (the same correction Adam uses) |
| (c) Uniform prior for **all** estimators (pseudo-count `n_0`) | Add `n_0` uniform pseudo-observations to cumulative and window too | Equal treatment; smooths early estimates | Adds a hyperparameter `n_0`; changes the approved cumulative/window definitions |

**Recommendation: (b)**, keeping (a) as an ablation. It makes the three estimators
differ *only* in how they weight past observations, which is exactly the adaptation
question. Zero estimates for unseen categories are then handled where they matter,
in the quality transform (`ε` in D-14).

**Evidence added 2026-11-01** (`results/m4_ewma_init/20261101-140141/`, tuning seeds 0–4;
`ewma_bias_corrected` is implemented as an extra option and is not the default):
- **At the same α** the versions give identical lag and post-shift error (the start has
  faded by the first switch). Bias correction lowers focused users' error over all steps
  (stable 0.125 vs 0.149) and raises spread users' (0.224 vs 0.207).
- **Each tuned on the pooled mix**, the selected α changes: 0.07 (uniform start, pooled
  TV 0.2216) vs **0.05** (bias-corrected, 0.2138, the lowest of all settings tried,
  including window(40) at 0.2164).
- **That lower α is a real trade-off:** bias-corrected α = 0.05 catches up more slowly
  (13.2 vs 9.3 steps, focused abrupt) but is more stable (0.103 vs 0.121 stable error;
  0.035 vs 0.050 noisy false switches).
Why the old α was higher: part of the tuning was compensating for the slow-fading start.
The recommendation stays (b), with `α* = 0.05`.

**M7 proposal for D-17 (2026-11-01), PENDING approval.** Notation: `R_t` = list of `K = 10`
items built from `ẑ_t`, sorted by score `s_i`; `T_t` = distinct items among `i_{t+1}..i_{t+h}`,
`h = 5`; `P_{t+1}` = exact item distribution from `GroundTruth`; `U_t` = items not clicked in
`i_1..i_t`. Lists are scored for `t = 20 … T−h` (burn-in 20).

| Metric | Definition | Role |
|---|---|---|
| **NDCG@K** | binary gain; `Σ_{p=1}^{K} 1[R_t(p) ∈ T_t] / log2(p+1)`, divided by the ideal for `min(|T_t|, K)` hits | **Primary relevance** |
| Recall@K | `|R_t ∩ T_t| / |T_t|` | Secondary |
| Expected utility EU@K | `Σ_{i∈R_t} P_{t+1}(i)` | Secondary; used to select `θ` |
| New-item NDCG, EU | as above, restricted to `T_t ∩ U_t` and `R_t ∩ U_t`; steps with no new target skipped | Repeat sensitivity |
| **Category coverage** | distinct categories in `R_t`, divided by `min(K, C)` | **Primary diversity** |
| ILD | mean over pairs in `R_t` of `1 − cos(φ_i, φ_j)` | Descriptive only (the DPP optimises this directly) |
| Seen share | fraction of `R_t` already clicked | Descriptive |
| Tracking error | `TV(ẑ_t, z_t)` | Estimation |
| Crossover lag | as implemented in M4, `run = 5`, censored at the horizon | Responsiveness |
| False-switch rate | as implemented in M4, burn-in 20 | Noise sensitivity |

Windows: post-shift = the 50 steps from each switch (abrupt 100; gradual midpoint 100;
recurrent 70 and 140). Aggregation: step → user → cell (κ, scenario) → seed. "Drift pooled" =
equal-weight mean over abrupt, gradual, recurrent and both κ.

**Three choices inside D-17, with evidence** (throwaway probe, tuning seeds, 30 users, every 5th
step, adaptive profile, Top-K vs `θ = 0.5`):
- **`h = 5` targets, not one next item.** Paired seed noise of the DPP/Top-K change:
  NDCG(`h=5`) 0.4–2.2%, Hit@K on one next item up to 16.9%. A 3% effect is detectable with the
  first and not reliably with the second.
- **Sort a DPP list by score before NDCG.** The DPP chooses a *set*. Greedy order is an
  artefact, and it cost the DPP 0–0.9% of NDCG in the probe. Sorting gives both selectors the
  same ranking rule.
- **Repeats: all targets as primary, new-item metrics as a reported sensitivity.** Repeats are
  33–64% of targets, which matches D-24 and the generator. The feared unfairness to DPP did
  not appear: on new items the DPP change was +0.9% to −4.3%, about the same as on all items.
  New-item EU is small (0.03–0.07).

**Recommendation:** approve as written.

---

## Group F — Raised preparing Milestone 6

### D-28 — Reuse `α*` or retune it for recommendation?
**Status:** APPROVED 2026-11-01: option (a), reuse `α* = 0.05`; retuned `α` is an exploratory sensitivity check. Raised 2026-11-01 (risk 18 in the log).

| Option | Pros | Cons |
|---|---|---|
| (a) **Reuse `α* = 0.05`** (tuned on estimation error) | The estimator is chosen independently of the selector, so H4–H6 compare one fixed estimator; simple; not tuned on a repeat-inflated metric | May not be the best `α` for lists |
| (b) Retune `α` per selector on a recommendation metric | Possibly better lists | More tuning; makes adaptation and selection harder to separate; the metric is affected by repeats (risk 19) |
**Recommendation: (a)**, with retuned `α` as an exploratory sensitivity check.

---

## Group G — Raised preparing Milestone 7

### D-29 — Rule for choosing `θ`
**Status:** APPROVED 2026-11-01: option (c). Raised 2026-11-01.

| Option | Pros | Cons |
|---|---|---|
| (a) Maximise relevance | Simple | Picks `θ ≈ 1`, meaning no DPP at all |
| (b) Maximise a weighted relevance + diversity score | One number | The weights are arbitrary and decide the outcome |
| (c) **Most diverse `θ` with an acceptable relevance loss** | Explicit trade-off; uses the already approved 3% SESOI | Depends on that threshold |

**Result (2026-11-01):** `θ = 0.5`, from `results/m7_theta/20261101-152049`: EU loss 2.8% (adaptive) and 1.0%
(static). `θ = 0.3` failed the rule (10.3% and 6.9%).

**Recommendation: (c).** On tuning seeds, pooled over all scenarios and both κ, choose the
smallest `θ` in {0.3, 0.5, 0.6, 0.7, 0.8, 0.9} whose EU@10 loss against same-profile Top-K is at
most 3%, for **both** the static and adaptive profiles. The chosen `θ` is used by all DPP
methods. The full `θ` curve is reported.

### D-30 — Confirmatory tests and H6(a) wording
**Status:** APPROVED 2026-11-01 as proposed. Fixes each hypothesis as one test before the freeze.
Each uses seed-level values (20 evaluation seeds), per D-21.

| Test | Quantity | Data |
|---|---|---|
| H1a | crossover lag: cumulative − EWMA | drift pooled |
| H1b | post-shift tracking error: cumulative − EWMA | drift pooled |
| H2 | tracking error: EWMA − cumulative | stable, both κ, `t > 20` |
| H3 | false-switch rate: EWMA − cumulative | noisy, both κ |
| H4a | post-shift NDCG@10: Adaptive Top-K − Static Top-K | drift pooled |
| H5a | category coverage: Adaptive DPP − Adaptive Top-K | all scenarios, both κ |
| H5b | relative NDCG cost of DPP (Adaptive): focused − spread | all scenarios |
| H6b | `(AdaDPP − StatDPP) − (AdaTop − StatTop)` on post-shift NDCG@10, two-sided | drift pooled |

Holm correction over these **8** tests. Two checks sit outside Holm: **H4b** (equivalence:
the whole 95% CI of the stable-scenario relative difference lies within ±3%) and **H6a**
(Adaptive DPP counts as dominated only if a method is better on both post-shift NDCG@10 and
category coverage, with both paired 95% CIs above zero). In H6a, "compared methods" means
Popularity, Static Top-K, Adaptive Top-K and Static DPP; **oracle methods are excluded**
because they are diagnostic ceilings (D-16). `hypothesis.md` would be amended to say this.
Per-scenario results are reported as descriptive.
**Recommendation:** approve as written.

---

## Revision history
- 2026-10-26 — File created in Milestone 0. All entries are PENDING or PRELIMINARY.
- 2026-10-26 — D-01 decided (fresh repo). D-02, D-03, D-05 and D-06 approved.
- 2026-10-26 — Milestone 1: added D-19 (calibration baseline), D-20 (hypotheses) and
  D-21 (statistical protocol). Citations in D-14/D-15 checked at metadata level (see
  `literature/literature_matrix.md`, A3 and A5). The trade-off parametrisation in D-14
  is still unverified until Chen et al. is read in full.
- 2026-10-26 — Milestone 2: D-08, D-09 and D-10 updated and proposed for approval;
  added D-22 … D-26. Specification in `SIMULATION_DESIGN.md`.
- 2026-10-28 — D-11 approved; implemented in M4. Added D-27 (EWMA initialisation).
- 2026-11-01 — D-27: evidence added, still pending. D-13: noted the conflict with D-24.
- 2026-11-01 — D-27 approved by the author (option b, `α* = 0.05`).
- 2026-11-01 — M5 proposals added to D-12 and D-13 (pending).
- 2026-11-01 — D-12 and D-13 approved by the author.
- 2026-11-01 — Papers read for M6. Proposals added to D-14, D-15, D-16; added D-28. D-19 is
  blocked until Steck (2018) is available in full.
- 2026-11-01 — D-14, D-15, D-16 and D-28 approved by the author.
- 2026-11-01 — M7 proposal added to D-17; added D-29 (`θ` rule) and D-30 (confirmatory tests).
- 2026-11-01 — D-17, D-29 and D-30 approved by the author.
- 2026-11-01 — `θ = 0.5` selected on tuning seeds under D-29. Added the M8 scale proposal to D-18.
- 2026-10-26 — Author approved D-08, D-09, D-10, the seed protocol (D-18, part),
  D-19 (option b), D-20, D-21 and D-22 … D-26, all as proposed. Still open: D-07
  (deadline) and D-11 … D-17 plus the scale part of D-18 (confirmed at their own
  milestones).
- 2026-11-02 — D-18 M8 scale approved by the author.
