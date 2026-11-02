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

Approved 2026-10-26 (Milestone 0), see `DECISIONS.md`:
- D-01: fresh repository; old GRU4Rec work discarded (decided by the author).
- D-02: `IMPLEMENTATION_LOG.md` is the single source of truth; nothing is deleted.
- D-03: fully synthetic catalogue and interactions for all main experiments; real
  data is optional at M9 and needs separate approval.
- D-04: open-loop evaluation on fixed sequences (from the master prompt).
- D-05: `adaptdiv/` package, JSON configs, pytest, `results/<run_id>/`.
- D-06: numpy, scipy, matplotlib, pandas, plus pytest (installed); no PyTorch.
Approved 2026-10-26 (after Milestones 1–2):
- D-08 … D-10: simplex interest vector; interaction model; five drift scenarios
  (`SIMULATION_DESIGN.md`).
- D-18 (seed protocol part): tuning seeds 0–4, evaluation seeds 100–119, SeedSequence
  streams, common random numbers across scenarios.
- D-19: adaptive calibration re-ranker as a secondary comparison.
- D-20: hypotheses H1–H6 (`hypothesis.md`). D-21: statistical protocol and SESOI.
- D-22 … D-26: independent centroids; `κ ∈ {0.1, 1.0}` with half-swap shift;
  repeats allowed; pooled-scenario tuning; training population for `r_i`.
Approved 2026-10-28 / 2026-11-01:
- D-11: cumulative mean, EWMA, sliding window.
- D-27: the proposed adaptive estimator is **bias-corrected EWMA, `α* = 0.05`**
  (`PROPOSED_ESTIMATOR`). The uniform-start EWMA (`α = 0.07`) is an ablation.
- D-12, D-13 (2026-11-01): `S = ΦΦᵀ + εI`; all items are candidates with no exclusion;
  within-category popularity as base relevance; global popularity only for the
  Popularity Top-K floor.
- D-14, D-15, D-16, D-28 (2026-11-01): log-score objective, Chen-style greedy on `S`, the
  method list, and reuse of `α* = 0.05`.
- D-17, D-29, D-30 (2026-11-01): metrics, `θ` rule, confirmatory tests. Selected `θ = 0.5`.
  The analysis is **frozen** in `hypothesis.md`.
Still pending: D-07 (deadline); the scale part of D-18 (M8). Scope from the master prompt:
controlled simulation where the true `z_t` is known; open-loop evaluation
on fixed sequences; real-data validation is optional and needs approval.

## 3. Project architecture and data flow [simulator part IMPL+TESTED (M3); rest PLANNED]

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

## 4. Repository structure [D-05 approved; files marked ✓ exist as of M3]

```
IRE Project/
├── README.md               setup, commands, overview
├── IMPLEMENTATION_LOG.md   this file
├── DECISIONS.md            decisions + approval status
├── SIMULATION_DESIGN.md    (M2) simulator specification: catalogue, z_t, drift, interaction model, seeds
├── hypothesis.md           approved hypotheses H1–H6 (frozen before evaluation runs)
├── literature/             (M1) literature matrix
├── adaptdiv/               Python package
│   ├── config.py      ✓    dataclasses + JSON load/save
│   ├── seeding.py     ✓    SeedSequence stream management
│   ├── catalogue.py   ✓    item categories, embeddings, popularity
│   ├── scenarios.py   ✓    z_t trajectory generators (stable/gradual/abrupt/noisy/recurrent)
│   ├── simulator.py   ✓    interaction generation; GroundTruth vs ObservedSequence
│   ├── estimators.py  ✓    cumulative mean, EWMA (bias-corrected + uniform start), sliding window
│   ├── estimation_metrics.py ✓ provisional tracking error, lag, false switches, noise floor
│   ├── relevance.py   ✓    base relevance r_i, plug-in scores, Top-K (quality transform g: M6)
│   ├── similarity.py  ✓    S = ΦΦᵀ (+ jitter), PSD checks
│   ├── dpp.py         ✓    greedy MAP (incremental Cholesky) + naive and brute-force references
│   ├── recommenders.py ✓   Top-K / DPP × static / adaptive / oracle
│   ├── metrics.py     ✓    relevance and diversity metrics (D-17)
│   ├── evaluation.py  ✓    per-seed evaluator, estimator records, phases
│   ├── stats.py       ✓    Wilcoxon, bootstrap CI, Holm
│   └── confirmatory.py ✓   the frozen D-30 analysis
├── experiments/
│   ├── configs/*.json
│   ├── sim_check.py   ✓    M3 simulator check (tuning seeds only)
│   └── run_experiment.py
├── analysis/               aggregation, statistics, plots
├── tests/         ✓        pytest (conftest, catalogue/seeding, scenarios, simulator)
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

Generative model (M2, APPROVED 2026-10-26, not yet implemented; full spec in `SIMULATION_DESIGN.md`):
- Catalogue: `μ_c = normalise(h_c)`, `h_c ~ N(0, I_d)`;
  `φ_i = normalise(μ_{c_i} + σ ε_i)`, `ε_i ~ N(0, I_d/d)`;
  popularity `π_i ∝ exp(s_pop ζ_i)`, `ζ_i ~ N(0,1)`, normalised within each category.
- Interests: `z_A ~ Dirichlet(κ 1_C)`; `z_B = P z_A`, where `P` is a half-swap
  permutation (top-`C/2` categories mapped onto bottom-`C/2` and vice versa).
  Trajectories: stable, abrupt, gradual
  `z_t = (1−w_t) z_A + w_t z_B`, recurrent, noisy.
- Interactions: `n_t ~ Bernoulli(η)`; `c_t ~ Cat(z_t)` if `n_t = 0`, else
  `Uniform(C)`; `P(i | c, u) ∝ π_i^γ exp(λ ŝ_i(u))` over items in `c`, where `ŝ_i(u)`
  is the standardised `⟨φ_i, u⟩`.
  Marginal: `P_t(i) = [(1−η) z_{t,c_i} + η/C] · P(i | c_i, u)`.
- Noise floor: `TV((1−η) z + η/C, z) = η · TV(z, uniform)`.

## 6. Implementation details and important functions

### Simulator (Milestone 3, 2026-10-26) [IMPL+TESTED]
Data flow for one seed: `simulator.generate(seed, cfg)` →
`catalogue.generate_catalogue` (stream `CATALOGUE`) → `train_item_counts`
(streams `TRAINING, j, …`) → for each `κ` index `k` and user `j`: `_user_params`
(stream `USERS, k, j, PARAMS`) and a `(3, T)` uniform bank (stream
`USERS, k, j, INTERACTIONS`) → for each scenario, `simulate_user` → `SimulatedUser(observed, truth)`.

| Module / function | Role |
|---|---|
| `config.SimulationConfig` | Frozen dataclasses with the approved defaults; `save_json`/`load_json` for exact run configs. `TUNING_SEEDS` = 0–4, `EVALUATION_SEEDS` = 100–119 |
| `seeding.stream(seed, *path)` | `SeedSequence(entropy=[PROJECT_TAG, seed], spawn_key=path)`. A stream depends only on (seed, path), so adding users or scenarios never changes existing draws |
| `catalogue.generate_catalogue` | Centroids, embeddings `φ_i`, within-category popularity `π_i`. Items are grouped by category (item `i` is in category `i // n_c`). Arrays are read-only |
| `catalogue.Catalogue.public()` | `PublicCatalogue` (categories and embeddings only), the view a recommender may receive |
| `scenarios.sample_base_interest` | `z_A ~ Dirichlet(κ1)` |
| `scenarios.half_swap_permutation` / `shifted_interest` | `z_B`: top-`C/2` categories ↔ bottom-`C/2` (random bijection); optional magnitude `m` |
| `scenarios.mixing_weights` / `trajectory` | `w_t` per scenario; `z_t = (1−w_t) z_A + w_t z_B`, with row `k` = step `t = k+1` |
| `simulator.item_conditional` | `(C, n_c)` matrix `P(i\|c,u) ∝ π_i^γ exp(λ ŝ_i(u))` (softmax per category, numerically stable) |
| `simulator.sample_interactions` | Inverse-CDF sampling from the uniform bank: noise flag, category (signal or uniform noise), then item within the category. Fully vectorised over `t` |
| `simulator.GroundTruth.item_distribution(k)` | Exact marginal `P_t(i)` for expected-utility metrics (M7) |
| `simulator.ObservedSequence` | **Only** the item-id array (read-only) plus `prefix(n)` |
| `simulator.train_item_counts` / `smoothed_popularity` | Training population counts (D-26); `r_i = (count_i+1)/(Σ+N)` |

Implementation choices made at this stage (routine, consistent with the approved design):
- The same uniform `U_cat[t]` serves both the signal draw (inverse CDF of `z_t`) and
  the noise draw (`⌊U_cat·C⌋`). Which one is used depends on the noise flag. This keeps
  three uniforms per step and preserves common random numbers.
- The order of draws inside the PARAMS stream is fixed (`z_A`, permutation, `u`), so
  changing one component later would change the others. This is documented, and the
  reproducibility tests would detect a change.
- Training users alternate `κ` levels (`j mod len(κ)`), use stable interests and the
  base noise rate, and live under a separate `TRAINING` stream.
- In `half_swap_permutation`, ties are broken by a stable sort, which is deterministic.

### Interest estimators (Milestone 4, 2026-10-28) [IMPL+TESTED]
`adaptdiv/estimators.py`. Input: the **observed** category sequence, obtained with
`categories_of(observed.items, public_catalogue.categories)`. Nothing from
`GroundTruth` is used. Output: `ẑ_0..ẑ_T` with shape `(B, T+1, C)`; row `t` uses
`c_1..c_t` only (tested: changing later observations leaves earlier estimates
unchanged). Batched over users with a loop over `T` (EWMA) or cumulative sums
(cumulative mean, window).

| Function | Definition |
|---|---|
| `cumulative_mean` | `ẑ_0` uniform; `ẑ_t = (1/t) Σ_{s≤t} x_s` |
| `ewma(alpha)` | `ẑ_0` uniform; `ẑ_t = (1−α) ẑ_{t−1} + α x_t` (as approved in D-11; see D-27) |
| `sliding_window(W)` | `ẑ_0` uniform; `ẑ_t = (1/min(t,W)) Σ_{s=max(1,t−W+1)}^{t} x_s` |
| `ewma_bias_corrected(alpha)` | Extra option added 2026-11-01 for D-27, not the default: `m_t = (1−α) m_{t−1} + α x_t`, `m_0 = 0`; `ẑ_t = m_t / (1−(1−α)^t)`; `ẑ_0` uniform |
| `ewma_mean_age`, `matched_window` | mean age `(1−α)/α`; equal-mean-age window `W = 2/α − 1` |
| `EstimatorSpec(kind, param)` | named, callable configuration used by experiments |

`adaptdiv/estimation_metrics.py` (**PROVISIONAL** until D-17 is approved at M7):
- `tracking_error`: `TV(ẑ_t, z_t)`, `t = 1..T`. Here `ẑ_t` includes the `t`-th
  observation, which was generated from `z_t`.
- `crossover_lag(start, run=5)`: the smallest `L ≥ 0` such that `ẑ_t` is closer (in TV)
  to the new interest than to the old one for `t = start+L … start+L+4`. If never,
  `L = T−start+1` and the case is flagged *censored*. `start` = 100 (abrupt), 100
  (gradual midpoint, where the truth is equidistant), and 70 and 140 (recurrent; the
  second switch is B→A).
- `false_switch_rate(burn_in=20)`: the fraction of steps `t > 20` at which
  `argmax ẑ_t ≠ argmax ẑ_{t−1}` while `argmax z_t = argmax z_{t−1}`.
- `noise_floor`: `η · TV(z_t, uniform)`, the error of a perfect tracker of the
  observed-category distribution.

### Evaluation framework (Milestone 7, 2026-11-01) [IMPL+TESTED]
Exact formulas: `DECISIONS.md` D-17 (see also §10).

| Module / function | Role |
|---|---|
| `metrics.py` | pure batched functions: `hits`, `n_targets`, `ndcg_at_k`, `recall_at_k`, `expected_utility`, `category_coverage`, `intra_list_diversity`. Repeated clicks in a window count once |
| `evaluation.build_profiles` | static, adaptive, uniform-start, window(40) from observed categories; true `z` only if passed explicitly, as `"oracle"` |
| `evaluation.method_lists` | lists for one user, sorted by score |
| `evaluation.list_metrics` | all D-17 metrics for one user; the only place lists meet `GroundTruth` and held-out clicks |
| `evaluation.estimator_records` | tracking error, lag, false switches for H1–H3 |
| `evaluation.evaluate`, `evaluate_seed` | seed-level records keyed `(κ, scenario, method, phase, metric)` |
| `stats.py` | two-sided Wilcoxon, percentile bootstrap CI, Holm |
| `confirmatory.py` | the 8 tests, H4b, H6a, verdicts. "Supported" = Holm p < 0.05 with the predicted sign; "practically meaningful" = whole CI beyond the SESOI |

Routine choice: per-step values are averaged per user, then per cell, then seed. NaN steps
(no new-item target) are skipped within a user.

### DPP and recommenders (Milestone 6, 2026-11-01) [IMPL+TESTED]
Objective (D-14): `F(Y) = θ Σ_{i∈Y} log s_i + (1−θ) log det S_Y`, with
`s_i = r_i (ẑ_{t,c_i} + 0.01)`. For `θ < 1`, `F(Y) = (1−θ) log det L_Y` with
`L = Diag(q) S Diag(q)`, `q_i = s_i^{θ/(2(1−θ))}` (tested). `θ = 1` is Top-K.
The linear trade-off in Chen et al. was rejected for our score scale; this form is our
adaptation and must be described as such.

| Function | Role |
|---|---|
| `dpp.greedy_map(log_s, S, theta, k)` | Chen et al. Algorithm 1 on `S`. Keeps `c_i` and `d_i²`. Step: pick `j = argmax θ log s_i + (1−θ) log d_i²`, then for every `i`: `e_i = (S_{ji} − ⟨c_j, c_i⟩)/d_j`, append `e_i` to `c_i`, `d_i² ← d_i² − e_i²`. Floors `d_i²` at 1e-12. Selected items are masked, so lists have no duplicates. `O(k²N)` |
| `dpp.objective` | `F(Y)` computed directly from `slogdet` |
| `dpp.naive_greedy_map`, `dpp.brute_force_map` | test references |
| `dpp.dpp_kernel` | explicit `L`, for validation only (powers of `s` can be badly scaled) |
| `recommenders.log_scores`, `select`, `recommendation_lists`, `popularity_list` | scores and lists; row `t` uses the profile for time `t` only |
| `recommenders.Method`, `standard_methods(theta)` | the D-16 method set |

Oracle profiles are passed in by experiment code only; `recommenders.py` never touches
`GroundTruth`. **Report wording:** "greedy approximate MAP inference".

### Relevance and similarity (Milestone 5, 2026-11-01) [IMPL+TESTED]
Inputs are observable only: training counts, public categories and embeddings, `ẑ_t`.

| Function | Definition |
|---|---|
| `relevance.global_popularity(counts)` | `(count_i+1)/(Σ_j count_j + N)`. Popularity Top-K floor only |
| `relevance.within_category_popularity(counts, categories, C)` | `r_i = (count_i+1)/(Σ_{j∈c_i} count_j + n_c)`; sums to 1 in each category |
| `relevance.plugin_scores(zhat, r, categories)` | `s_i(t) = r_i · ẑ_{t,c_i}`, batched. Sums to 1 over items, so it is an estimated next-item distribution |
| `relevance.top_k(scores, k)` | best `k` items, deterministic tie-break (lowest index) |
| `similarity.similarity_matrix(embeddings, jitter=1e-6)` | `S = ΦΦᵀ + εI` |
| `similarity.min_eigenvalue(S)` | verifies positive definiteness |

**Candidates:** all `N` items for every method, so there is no separate candidate
generator. **Timing rule:** the list for step `t+1` uses `ẑ_t` only.
The quality transform `g` and DPP kernel are M6 (D-14).

## 7. Data sources, preprocessing, data generation [IMPL+TESTED 2026-10-26 (M3)]
Fully synthetic catalogue and interactions (D-03, approved 2026-10-26).
**Approved simulator design (M2, 2026-10-26): see `SIMULATION_DESIGN.md`** (catalogue §2,
interests §3, scenarios §4, interaction model §5, firewall §6, seeds §7, scale §8,
limitations §10). Proposed defaults: `C=10`, `N=500`, `d=32`, `σ=1`, `s_pop=1`,
`γ=1`, `λ=0.5`, `η=0.05` (0.3 in the noisy scenario), `κ ∈ {0.1, 1.0}`, `T=200`,
`τ=100`, gradual ramp 80–120, recurrent switches at 70/140, `U=100` per (seed,
scenario, `κ`), training population 1,000 × 50 per seed.

## 8. Experimental configurations and hyperparameters [PLANNED — D-18]
Preliminary: `C=10, N=500, d=32, K=10, T=200, U=100`; tuning seeds are kept separate
from evaluation seeds. Nothing has been fixed yet.
M2 proposal: tuning seeds 0–4, evaluation seeds 100–119. One hyperparameter setting
per method, tuned on a pooled scenario mix (D-25). Common random numbers across
scenarios (`SIMULATION_DESIGN.md` §7).

**2026-11-02 (M8), as run:** `U = 100`, 1,000 training users, 20 evaluation seeds, every step,
`K = 10`, `h = 5`, `α = 0.05`, `θ = 0.5` (D-18 scale approved by the author).

## 9. Baselines and ablation studies [PLANNED — D-16]
**2026-11-02 (M8), as run [IMPL+TESTED]:** the 2×2 methods, Popularity, Oracle Top-K and Oracle DPP,
plus ablations uniform-start EWMA and window(40), each with Top-K and DPP. Calibration (D-19) was not
run: Steck (2018) is still unavailable. MMR stays dropped (D-16). Sensitivity sweeps were not run at M8.

A 2×2 design of {static, adaptive} × {Top-K, DPP}, plus popularity, plus an oracle-z
diagnostic ceiling. Estimator variants: EWMA vs sliding window. Sensitivities: `α`/`W`,
noise `η`, interest concentration `κ`, drift speed `W`, and the DPP trade-off.

## 10. Evaluation metrics and calculation procedures [PLANNED — D-17]
Exact formulas will be written here at M7 and hand-checked in tests.
**2026-11-01:** approved formulas are in `DECISIONS.md` D-17, implemented in `adaptdiv/metrics.py`
and `adaptdiv/evaluation.py`, hand-checked in `tests/test_metrics_evaluation.py`.

## 11. Commands used to run experiments
Run from the repository root (`ire-adaptive-dpp-rec/`):
```bash
python -m pytest -q                      # full test suite
python experiments/sim_check.py          # M3 simulator check on tuning seeds 0-4
python experiments/estimator_study.py    # M4 estimator tuning + comparison on tuning seeds 0-4 (~12 s)
python experiments/ewma_init_comparison.py  # D-27 evidence: uniform-start vs bias-corrected EWMA, tuning seeds
python experiments/relevance_check.py    # M5 Top-K scoring check on tuning seeds 0-4 (~30 s)
python experiments/dpp_check.py          # M6 greedy quality, speed, theta sweep on tuning seeds 0-4 (~3 min)
python experiments/select_theta.py       # M7 D-29 theta selection + pipeline rehearsal, tuning seeds (~7 min, 5 processes)
python experiments/main_experiment.py --dry-run --out <dir>  # M8 rehearsal: tuning seeds, tiny scale (~2 s)
python experiments/main_experiment.py    # M8 confirmatory run, EVALUATION seeds (~14 min, 5 processes)
```
`sim_check.py` refuses evaluation seeds (≥ 100). Output goes to
`results/m3_sim_check/<run_id>/` (`config.json`, `summary.json`,
`environment.json`, `trajectories.png`).

## 12. Environment and dependency information
Recorded 2026-10-26 (Milestone 0 inspection):
- Windows 11 Pro 10.0.26200; 20 logical CPUs; 15.7 GB RAM; no GPU used.
- Python 3.12.10; numpy 2.5.2; scipy 1.18.0; pandas 3.0.5; matplotlib 3.11.1.
- torch 2.13.0+cpu is installed (from the earlier phase) and is **not needed** by the
  new design.
- ~~pytest is not installed~~ → pytest 9.1.1 installed 2026-10-26 (D-06 approved).
- ~~Git repository; remote `origin = https://github.com/Shambo-Roy/ire-adaptive-session-rec.git`.~~
  (Superseded 2026-10-26: the author deleted that repository.)
- Git repository, freshly initialised 2026-10-26, branch `main`; remote
  `origin = https://github.com/Shambo-Roy/ire-adaptive-dpp-rec` (public).

## 13. Test cases and outcomes

### Milestone 7 (2026-11-01): `python -m pytest -q` → **159 passed in 6.01 s**
- `test_metrics_evaluation.py` (20): hand-calculated NDCG, recall, EU, coverage, ILD; repeated
  targets count once; **independent slow recomputation** of all metrics for real lists; DPP lists
  sorted by score; records complete; `θ = 1` DPP gives identical records to Top-K; window alignment.
- **Leakage:** import check that estimators, relevance, similarity, DPP and recommenders can't
  reach the simulator; no oracle profile unless passed; replacing all clicks after `t` leaves lists
  up to `t` unchanged; oracle row `t` is the interest at step `t+1`.
- `test_confirmatory.py` (11): Holm and exact Wilcoxon by hand; planted effects found with the right
  direction; **false-positive rate under 10% over 200 null tables**; wrong direction not supported;
  a small effect is significant but not practically meaningful; H4b; H6a finds dominators and
  ignores oracles.
- A first version of the null test failed: it required no false positive in 5 tables, which is not
  a valid expectation at a 5% error rate. It was replaced by the rate check. One meaningless
  assertion in the coverage test was also removed.

### Milestone 6 (2026-11-01): `python -m pytest -q` → **128 passed in 0.99 s**
`test_dpp_recommenders.py` (44 new, counting parametrised cases): hand example (`θ=0.5`
avoids the near-duplicate; `θ=1` doesn't); `θ=0` ignores scores; **fast greedy picks exactly
the same items as naive greedy** (24 cases); greedy never beats brute force and stays close;
`θ=1` equals Top-K; `F(Y) = (1−θ) log det L_Y`; `L` symmetric positive definite; lists stay
distinct when `k` exceeds the rank; invalid arguments; zero interest gives finite scores;
list shape, step selection and **no look-ahead**; DPP covers at least as many categories as
Top-K for uniform interest; D-16 method set; popularity list. One test written during this
milestone was removed before counting because it only compared a determinant with itself.

### Milestone 5 (2026-11-01): `python -m pytest -q` → **84 passed in 0.98 s**
`test_relevance_similarity.py` (13 new): hand-calculated global and within-category
popularity; positivity, sums to 1 per category, empty-category behaviour; hand-calculated
plug-in scores; scores form a distribution and batching agrees; a zero estimate gives a
zero score (documents what the DPP quality transform must handle); `top_k` order, ties and
batching; `S` symmetric with diagonal `1+ε`; positive definite with jitter; `ΦΦᵀ` PSD with
rank exactly 32; embeddings not modified; a duplicated item gives determinant 0.

### Milestone 4 (2026-10-28, extended 2026-11-01): `python -m pytest -q` → **71 passed in 0.78 s**
65 tests passed at the end of M4 on 2026-10-28; 6 were added on 2026-11-01 for the
bias-corrected EWMA.
| File | What is checked |
|---|---|
| `test_estimators.py` | hand-calculated EWMA, bias-corrected EWMA, cumulative-mean and window values; unrolled and normalised-average formulas; limits (`α=1`, `W=1`, `W≥T`, `α→0` gives the cumulative mean); the two EWMA versions agree once the start has faded; simplex and uniform `ẑ_0`; **no look-ahead**; batch = single; long-run convergence; one noisy click moves EWMA by exactly `α`; matched memory; invalid arguments |
| `test_estimation_metrics.py` | TV; `ẑ_t`/`z_t` alignment; noise floor; crossover lag (hand example, censoring, batching); false-switch rate (hand example, burn-in, a true switch is not false) |

### Milestone 3 (2026-10-26): `python -m pytest -q` → **39 passed in 0.66 s** (pytest 9.1.1)
| File | What is checked |
|---|---|
| `test_catalogue_and_seeding.py` (8) | shapes and grouping; unit norms; popularity positive and summing to 1 per category; within/across cosine ≈ 0.50/0.00 (10 catalogues, tolerance 0.03); public view hides popularity and centroids; read-only arrays; same seed gives the same catalogue and different seeds differ; streams depend only on (seed, path); JSON config round-trip |
| `test_scenarios.py` (13) | `z_A` on the simplex (both `κ`); E[max z] ≈ 0.665/0.293 (5,000 draws, tol 0.02); half-swap is a permutation that moves every top category into the bottom half and keeps the sorted values; mean shift TV ≈ 0.99/0.65 (3,000 draws, tol 0.03); partial-shift magnitude; every scenario on the simplex; exact switch points (abrupt 100, recurrent 70/140, gradual 80–120 with `w=0.5` at t=100); noise rates |
| `test_simulator.py` (18) | conditional rows are distributions; `λ=0` reduces to `π^γ`; **200,000-step stable sequence**: noise rate within 0.005 of `η`, category-frequency TV < 0.01, item-frequency TV < 0.03 vs the model; exact marginal `P_t(i)`; item ids consistent with categories; **common random numbers** (stable = abrupt for t<100, = recurrent for t<70, = gradual for t≤80; noisy differs only where its noise flag differs); post-switch interactions follow `z_B` (pooled TV gap > 0.2); reproducible `generate`; adding users leaves existing users unchanged; different seeds differ; params shared across scenarios; **firewall** (`ObservedSequence` has only `items`, integer and read-only; ground-truth arrays read-only); training counts sum correctly and are independent of the number of evaluation users |

Statistical tests use fixed seeds, so they are deterministic. Their tolerances were set
before running, from the expected Monte Carlo error.

## 14. Experiment results
*No confirmatory experiments have been run. The checks below use tuning seeds only and
are not hypothesis tests.*

### M7 `θ` selection (tuning seeds 0–4, 2026-11-01)
- **Run:** `results/m7_theta/20261101-152049`. Full scale (100 users, every step, `t = 20…195`), 15 methods, 5 processes,
  392 s.
- **D-29 rule** (EU@10 loss vs same-profile Top-K, pooled; negative = gain; coverage is the
  category-coverage metric):

| `θ` | static loss | adaptive loss | static coverage | adaptive coverage |
|---|---|---|---|---|
| 0.9 | −0.2% | 0.0% | 0.364 | 0.332 |
| 0.8 | −0.3% | 0.1% | 0.378 | 0.345 |
| 0.7 | −0.3% | 0.5% | 0.394 | 0.360 |
| 0.6 | 0.2% | 1.3% | 0.414 | 0.380 |
| **0.5** | 1.0% | 2.8% | 0.440 | 0.406 |
| 0.3 | 6.9% | 10.3% | 0.523 | 0.490 |

  **Selected `θ = 0.5`**, the smallest with loss ≤ 3% for both profiles. It sits close to the limit.
- **Pipeline rehearsal** (`rehearsal.json`): the frozen analysis ran end to end on 5 tuning seeds.
  With 5 seeds the smallest possible Wilcoxon p is 0.0625, so nothing can be significant. **Not a
  result.** Its purpose was to prove that the code runs.

### M6 DPP check (exploratory, tuning seeds 0–4, 2026-11-01)
- **Run:** `results/m6_dpp/20261101-150200`. Subsample: first 30 users per cell, every 5th step after step 20;
  `K = 10`; runtime 167 s. Plot: `tradeoff.png`. Coverage metric as in M5 (provisional).
  Stable: all sampled steps. Abrupt: 50 steps after the switch.
- **Greedy vs exact** (200 random 14-item instances, `k = 5`, real embeddings): exact optimum
  found 95.0% / 99.5% / 100% of the time at `θ` = 0.3 / 0.6 / 0.8; largest objective gap
  0.069 / 0.007 / 0. Tiny instances only: this does not bound the gap at `N = 500`.
- **Speed:** 0.12–0.18 ms per list at `N = 500`, `K = 10`.
- **Sweep** (coverage / categories per list):

| κ, scenario | θ | static | adaptive | oracle-z |
|---|---|---|---|---|
| focused, stable | 1.0 | 0.376 / 1.9 | 0.373 / 1.9 | 0.377 / 1.9 |
| | 0.5 | 0.363 / 2.3 | 0.361 / 2.3 | 0.366 / 2.2 |
| | 0.3 | 0.335 / 2.9 | 0.332 / 2.9 | 0.340 / 2.7 |
| focused, abrupt | 1.0 | 0.062 / 2.4 | 0.277 / 2.5 | 0.379 / 1.9 |
| | 0.5 | 0.094 / 3.2 | 0.266 / 3.2 | 0.369 / 2.2 |
| | 0.3 | 0.114 / 3.9 | 0.243 / 4.0 | 0.341 / 2.6 |
| spread, stable | 1.0 | 0.191 / 4.2 | 0.185 / 4.0 | 0.197 / 4.2 |
| | 0.5 | 0.185 / 5.2 | 0.180 / 5.0 | 0.190 / 5.3 |
| | 0.3 | 0.171 / 6.2 | 0.168 / 5.9 | 0.176 / 6.2 |
| spread, abrupt | 1.0 | 0.086 / 4.9 | 0.156 / 4.7 | 0.205 / 4.2 |
| | 0.5 | 0.099 / 6.1 | 0.154 / 5.9 | 0.198 / 5.3 |
| | 0.3 | 0.108 / 7.1 | 0.146 / 6.8 | 0.182 / 6.2 |

  Full results, including gradual, recurrent, noisy, ILD and seen share, are in `check.json`.
  Popularity floor: 0.13–0.14. Seed-to-seed SD of adaptive coverage at `θ = 0.6` after the
  shift: 0.018 (focused), 0.028 (spread).
- **Reading (directional only):**
  - For the static profile after a shift, more diversity **raises** coverage (0.062 → 0.114
    focused), because a diverse list hedges against stale interests. This is the
    *substitution* direction of H6(b): the adaptation gain at `θ = 0.5` is 0.172 vs 0.215 at
    Top-K for focused users, and 0.055 vs 0.070 for spread users.
  - The adaptive profile loses little down to `θ ≈ 0.5`, while categories rise.
  - Relevance cost at `θ = 0.5` (stable): 3.2% focused, 2.7% spread; after a shift 4.0% and
    1.3%. That is the H5 direction, but the stable gap is small.
  - Adaptive DPP is below the static DPP on stable interests, as H2 would imply.

### M5 scoring check (exploratory, tuning seeds 0–4; not a hypothesis test)
- **Run:** `results/m5_relevance/20261101-142207` (Top-K, `K = 10`, runtime about 31 s; 5 seeds × 2 `κ` × 5 scenarios ×
  100 users × 200 steps).
- **Measure:** coverage `Σ_{i∈R_t} P_{t+1}(i)`, the true next-item probability covered by
  the list. Provisional: formal metrics are fixed at M7.
- **Profiles:** popularity (no profile, global `r`), static (cumulative mean), adaptive
  (bias-corrected EWMA, `α = 0.05`), oracle-z (true `z_{t+1}`, a diagnostic ceiling only).
- Stable and noisy: mean over steps 21–200. Drift scenarios: 50 steps after each switch.

| κ | scenario | popularity | static | adaptive | oracle-z | listed items already clicked (adaptive) |
|---|---|---|---|---|---|---|
| 0.1 | stable | 0.141 | 0.372 | 0.369 | 0.374 | 0.87 |
| 0.1 | abrupt | 0.133 | 0.061 | 0.276 | 0.370 | 0.78 |
| 0.1 | gradual | 0.134 | 0.098 | 0.289 | 0.339 | 0.78 |
| 0.1 | recurrent | 0.137 | 0.192 | 0.281 | 0.372 | 0.79 |
| 0.1 | noisy | 0.140 | 0.291 | 0.287 | 0.294 | 0.83 |
| 1.0 | stable | 0.139 | 0.195 | 0.188 | 0.201 | 0.76 |
| 1.0 | abrupt | 0.140 | 0.082 | 0.155 | 0.204 | 0.69 |
| 1.0 | gradual | 0.139 | 0.096 | 0.159 | 0.191 | 0.69 |
| 1.0 | recurrent | 0.139 | 0.129 | 0.155 | 0.202 | 0.69 |
| 1.0 | noisy | 0.139 | 0.169 | 0.161 | 0.175 | 0.73 |

- **Kernel:** `S` for 500 items builds in about 0.01 s; minimum eigenvalue about 1e-6
  (the jitter), as expected.
- **Reading:** the ordering is sensible: oracle-z is highest; when stable the static
  profile is slightly better than adaptive; after a shift adaptive is well above static,
  and static falls below even popularity. Directional only, in the direction of H4.
- **Repeat warning:** 69–87% of listed items were already clicked by that user. See risk 15.

### D-27 comparison: EWMA uniform start vs bias-corrected (exploratory, tuning seeds 0–4, 2026-11-01)
- **Run:** `results/m4_ewma_init/20261101-140141/`, same sequences and objective as the
  M4 study below.
- **Pooled mean TV by α** (0.02 / 0.03 / 0.05 / 0.07 / 0.1 / 0.15 / 0.2 / 0.3 / 0.5):
  uniform start 0.2818 / 0.2474 / 0.2230 / **0.2216** / 0.2330 / 0.2612 / 0.2906 / 0.3452 / 0.4384;
  bias-corrected 0.2339 / 0.2210 / **0.2138** / 0.2187 / 0.2339 / 0.2640 / 0.2937 / 0.3480 / 0.4402.
- Selected α: 0.07 (uniform start), 0.05 (bias-corrected).

| κ | scenario | EWMA(0.07), uniform | bias-corrected (0.07) | bias-corrected (0.05) |
|---|---|---|---|---|
| 0.1 | stable: error, steps 21–200 / false-switch | 0.121 / 0.044 | 0.115 / 0.044 | 0.103 / 0.032 |
| 0.1 | abrupt: post / lag | 0.317 / 9.3 | 0.317 / 9.3 | 0.384 / 13.2 |
| 0.1 | noisy: error / false-switch | 0.257 / 0.050 | 0.251 / 0.050 | 0.242 / 0.035 |
| 1.0 | stable: error / false-switch | 0.202 / 0.138 | 0.205 / 0.138 | 0.177 / 0.102 |
| 1.0 | abrupt: post / lag | 0.283 / 9.9 | 0.283 / 9.9 | 0.295 / 14.0 |
| 1.0 | noisy: error / false-switch | 0.234 / 0.161 | 0.236 / 0.161 | 0.211 / 0.119 |

- **Reading:** at the same α, lag and post-shift error are identical, because the start
  has faded by the first switch. The difference is early error, and the tuned α: the
  uniform start pushed tuning towards a larger α, so correcting it gives a slower but
  steadier estimator.

### M4 estimator study (exploratory, tuning seeds 0–4; not a hypothesis test) [IMPL+TESTED]
- **Run:** `results/m4_estimators/20261028-121944` (default `SimulationConfig`; 2 `κ` × 5 scenarios × 100 users ×
  200 steps per seed; runtime about 12 s). The EWMA uses the uniform start approved in
  D-11 (see D-27).
- **Tuning (D-25: pooled over all scenarios, both `κ`, all steps, 5 seeds)**, mean
  `TV(ẑ_t, z_t)`:

| estimator | pooled mean TV |
|---|---|
| cumulative | 0.2808 |
| EWMA α = 0.02 / 0.03 / 0.05 / **0.07** / 0.1 / 0.15 / 0.2 / 0.3 / 0.5 | 0.2818 / 0.2474 / 0.2230 / **0.2216** / 0.2330 / 0.2612 / 0.2906 / 0.3452 / 0.4384 |
| window W = 3 / 5 / 7 / 10 / 15 / 20 / 30 / **40** / 60 / 100 | 0.4663 / 0.3787 / 0.3301 / 0.2878 / 0.2512 / 0.2328 / 0.2182 / **0.2164** / 0.2280 / 0.2617 |

  Selected: **α\* = 0.07**, **W\* = 40**. The window with the same mean age as
  α\* is W = 28.

- **Per scenario** (mean over 5 tuning seeds; error = mean TV over steps 21–200;
  pre/post = mean TV over steps 21…switch−1 / the 50 steps after each switch; lag =
  mean crossover lag in steps, with the censored fraction in brackets; seed-to-seed SD
  of the error ≤ 0.006 throughout):

| κ | scenario | cumulative | EWMA(0.07) | window(40) | window(28) |
|---|---|---|---|---|---|
| 0.1 | stable: error / false-switch rate | 0.076 / 0.009 | 0.121 / 0.044 | 0.101 / 0.019 | 0.115 / 0.027 |
| 0.1 | abrupt: post / lag | 0.791 / 98.7 (69% cens.) | 0.317 / 9.3 | 0.425 / 19.5 | 0.334 / 13.5 |
| 0.1 | gradual: post / lag | 0.685 / 98.0 (67%) | 0.254 / 12.7 | 0.341 / 20.1 | 0.259 / 14.1 |
| 0.1 | recurrent: post / lag | 0.587 / 53.7 (31%) | 0.314 / 9.2 | 0.424 / 19.5 | 0.333 / 13.5 |
| 0.1 | noisy: error / false-switch | 0.228 / 0.009 | 0.257 / 0.050 | 0.242 / 0.023 | 0.251 / 0.032 |
| 1.0 | stable: error / false-switch | 0.115 / 0.024 | 0.202 / 0.138 | 0.172 / 0.069 | 0.202 / 0.095 |
| 1.0 | abrupt: post / lag | 0.513 / 93.2 (56%) | 0.283 / 9.9 | 0.324 / 19.6 | 0.301 / 13.7 |
| 1.0 | gradual: post / lag | 0.446 / 94.0 (60%) | 0.247 / 13.2 | 0.268 / 20.2 | 0.252 / 14.5 |
| 1.0 | recurrent: post / lag | 0.383 / 52.1 (27%) | 0.282 / 9.9 | 0.324 / 19.8 | 0.301 / 13.7 |
| 1.0 | noisy: error / false-switch | 0.160 / 0.026 | 0.234 / 0.161 | 0.207 / 0.082 | 0.235 / 0.107 |

  Noise floors: `κ=0.1` 0.035 (noisy 0.212); `κ=1.0` 0.017 (noisy 0.103).
- **α trade-off (EWMA, abrupt and noisy, κ = 0.1):** as α goes 0.02 → 0.5, the
  crossover lag falls 30.5 → 0.7 steps, while the noisy-scenario false-switch rate
  rises 0.017 → 0.680. The stable error is U-shaped (0.173 at α=0.02, minimum 0.117 at
  α=0.05, 0.291 at α=0.5). The rise at small α comes from the uniform start (D-27).
- **Oracle (per-scenario) α**, for the "oracle-tuned" sensitivity of D-25: `κ=0.1`:
  stable 0.07, abrupt 0.1, gradual 0.1, recurrent 0.15, noisy 0.07; `κ=1.0`: stable
  0.02, abrupt/gradual/recurrent 0.05, noisy 0.02.
- **Plots:** `tuning_curves.png`, `error_over_time.png` (error curves per scenario and
  `κ` with the noise floor).
- **Reading (exploratory, tuning seeds; directional only):** the patterns are in the
  directions predicted by H1 (EWMA recovers about 10× faster than the cumulative mean,
  which is mostly censored), H2 (the cumulative mean is more accurate in the stable
  scenario) and H3 (EWMA has about 5× more false switches). These are *not* tests of
  H1–H3. The confirmatory tests use the evaluation seeds at M8, after the freeze.
- **Matched memory is not matched lag:** EWMA(0.07) and window(28) have the same mean
  observation age (about 13 steps), but EWMA crosses over faster (9.3 vs 13.5 steps).
  EWMA puts more weight on the very latest observations, while the window weights its
  W observations equally.

### M3 simulator verification run (not a hypothesis test) [IMPL+TESTED]
- **Run:** `results/m3_sim_check/20261026-192349` (default `SimulationConfig`, tuning seeds 0–4, 2026-10-26).
- **Scale per seed:** 200,000 evaluation interactions (2 `κ` × 5 scenarios × 100 users
  × 200) plus 50,000 training interactions.
- **Generation time:** 0.22–0.25 s per seed (Windows 11, 20 logical CPUs, NumPy
  2.5.2, single process). Full 20-seed generation would take about 5 s, so generation is
  not a bottleneck.
- **Catalogue:** mean cosine within category 0.496–0.505, across −0.015…0.018. No
  item had zero training count.
- **Means over seeds** (`maxz` = mean max component of `z_A`; `noise` = observed
  noise-flag rate; `repeat` = fraction of interactions repeating an item the user
  already clicked):

| κ | scenario | shift TV | maxz | noise | repeat |
|---|---|---|---|---|---|
| 0.1 | stable / abrupt / gradual / recurrent / noisy | 0.991 | 0.657 | 0.050 (noisy 0.300) | 0.61 / 0.48 / 0.48 / 0.49 / 0.49 |
| 1.0 | stable / abrupt / gradual / recurrent / noisy | 0.642 | 0.289 | 0.049 (noisy 0.299) | 0.41 / 0.35 / 0.35 / 0.35 / 0.37 |

- **Windowed tracking** (TV between the category frequency of 20 consecutive
  interactions and the window-average of `(1−η) z_t + η/C`): flat across all ten
  windows in every scenario (focused 0.14–0.15, spread 0.23–0.24; noisy 0.22 /
  0.25–0.26). The generator therefore follows `z_t` with no lag of its own. The level
  is the sampling noise of 20 draws, which is larger for higher-entropy distributions.
  This is why the focused gradual windows covering the ramp (t = 81–120) rise to
  0.19–0.20: mid-ramp, `z_t` is a mixture of two focused vectors and so has higher
  entropy.
- **Within-category concentration** (seed 0, focused users, separate quick check):
  effective number of items `exp(H(P(·|c,u)))` = 29.2 of 50; the top-5 items take 40%
  of clicks in a category.
- `trajectories.png`: one example user per scenario and `κ`. True `z_t` for the top
  categories of `z_A` and `z_B` against the rolling 20-interaction observed frequency.

### Milestone 8 — confirmatory run on evaluation seeds (2026-11-02) [MEASURED]
Run: `results/m8_main/20261102-112410/`. Seeds 100–119, `U = 100` per cell, every step, `θ = 0.5`,
`α = 0.05`, git HEAD `f0fc858`, `adaptdiv/` unchanged from HEAD. 812 s with 5 processes. 29,440
records, no NaN, no missing cell. This was the first and only run on these seeds.

**Holm family (8 tests).** `1.91e-06` is the smallest p a two-sided Wilcoxon can give with 20 seeds.

| Test | Mean difference | 95% CI | Holm p | Verdict |
|---|---|---|---|---|
| H1a lag, cumulative − EWMA | +67.3 steps | [67.0, 67.6] | 1.5e-05 | Supported, beyond SESOI |
| H1b post-shift TV, cumulative − EWMA | +0.248 | [0.247, 0.249] | 1.5e-05 | Supported, beyond SESOI |
| H2 stable TV, EWMA − cumulative | +0.045 | [0.045, 0.046] | 1.5e-05 | Supported, beyond SESOI |
| H3 noisy false switches, EWMA − cumulative | +0.057 | [0.055, 0.058] | 1.5e-05 | Supported, no SESOI |
| H4a post-shift NDCG, Adaptive − Static Top-K | +0.101 (+131% relative) | [122%, 140%] | 1.5e-05 | Supported, beyond SESOI |
| H5a coverage, Adaptive DPP − Adaptive Top-K | +0.089 | [0.086, 0.092] | 1.5e-05 | Supported, no SESOI |
| H5b relative NDCG cost, focused − spread | +0.0038 | [0.0012, 0.0064] | 0.024 | Supported, no SESOI; very small |
| H6b interaction on post-shift NDCG | −0.0106 | [−0.0111, −0.0100] | 1.5e-05 | Significant: substitution |

**Separate checks.** H4b: the stable relative NDCG difference has CI [−1.9%, −1.6%], inside ±3%, so
it meets the equivalence rule, although Adaptive Top-K is measurably lower. H6a: Adaptive DPP is
not dominated; no compared method beats it on both axes.

**Pooled means.**

| Method | Post-shift NDCG | Post-shift EU | Post-shift coverage | Overall NDCG | Overall coverage | Overall seen share |
|---|---|---|---|---|---|---|
| Popularity | 0.1026 | 0.1281 | 0.745 | 0.1029 | 0.745 | 0.490 |
| Static Top-K | 0.0787 | 0.1027 | 0.358 | 0.1774 | 0.328 | 0.805 |
| Adaptive Top-K | 0.1794 | 0.2129 | 0.341 | 0.2086 | 0.299 | 0.756 |
| Static DPP | 0.0861 | 0.1175 | 0.463 | 0.1762 | 0.421 | 0.787 |
| Adaptive DPP | 0.1762 | 0.2064 | 0.444 | 0.2049 | 0.388 | 0.744 |
| Oracle Top-K (diagnostic) | 0.2348 | 0.2736 | 0.294 | 0.2281 | 0.287 | 0.720 |
| Oracle DPP (diagnostic) | 0.2304 | 0.2648 | 0.376 | 0.2241 | 0.366 | 0.710 |

Seed-to-seed standard deviation of post-shift NDCG for Adaptive DPP: 0.012.

**What the numbers mean.**
- Adaptation gives the large gain. The DPP's change is small and has opposite signs: +0.0074
  post-shift NDCG with the static profile, −0.0032 with the adaptive profile. That gap is H6b's
  substitution result.
- Adaptive DPP against Adaptive Top-K: NDCG cost 1.8% and coverage +0.089 (0.299 → 0.388). Overall
  EU loss is 3.04%, just above the 3% limit that `θ` met on tuning seeds (2.8%). The tuning rule is
  not re-applied here; this is reported as the out-of-sample value. The static profile's loss is 1.05%.
- H5b holds in direction but is not a strong interaction: cost 1.93% focused, 1.55% spread; 14 of
  20 seeds positive.
- Estimation matters more than selection here: Oracle Top-K exceeds Adaptive Top-K by 0.055
  post-shift NDCG, about 17 times the DPP's cost of 0.0032.
- H2 and H3 are the predicted costs of adaptation and both appeared. In the noisy scenario EWMA's
  false-switch rate is 0.074 against 0.017.

**Ablations (descriptive, outside Holm; variant − proposed, drift pooled, post-shift).**

| Variant | NDCG Top-K | NDCG DPP | Coverage Top-K | Coverage DPP |
|---|---|---|---|---|
| Uniform-start EWMA (`α = 0.07`) | +0.0114 [0.0111, 0.0118] | +0.0113 [0.0110, 0.0117] | −0.022 | −0.030 |
| Window(40) | −0.0179 [−0.0184, −0.0175] | −0.0154 [−0.0159, −0.0149] | −0.005 | −0.011 |

**A result against the proposed configuration.** The uniform-start ablation beats the proposed
estimator on post-shift NDCG by about 6%, while losing coverage. Overall the NDCG gap is only
+0.0009. This ablation changes two things at once (`α` 0.07 vs 0.05, and the start), so it cannot
say which one matters; `α` is the likely reason, but that is untested. `α = 0.05` was chosen for
tracking error on tuning seeds (D-27), not for relevance. It stays fixed: the evaluation seeds
have now been seen, so any new `α` would need fresh seeds and would be labelled exploratory.

**Checks on this run** (scratchpad script, not in the repo).
- Seed 100 rerun alone in one process: all 1,472 values identical to the pooled run.
- H5b and H6b recomputed from `records.csv` with scipy directly: same means and p-values.
- Oracle Top-K has the highest EU among Top-K methods in all 10 cells.
- **Lag censoring:** the cumulative mean never crosses within the horizon for 64% of abrupt and
  gradual cases and 29% of recurrent ones. EWMA is never censored. Censored lags are recorded at
  the horizon, so H1a's 67 steps understates the true gap.

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

- **2026-10-26 (M1):** the deadline was requested twice, but both replies had
  blank date fields. D-07 stays open. M1 went ahead because it doesn't depend on the
  deadline.
- **2026-10-26 (M1):** Steck (2018), *Calibrated Recommendations*: the primary text
  was not retrieved by search. Its claims are recorded at level S (secondary)
  until it is read in full.

- **2026-10-26 (M2):** design-stage Monte Carlo checks (throwaway scratchpad scripts,
  not in the repository) changed two preliminary ideas:
  (i) the proposed "swap top-2 categories" drift gave a mean TV shift of only 0.36
  for spread users (5th percentile 0.16), too weak to test adaptation. It was replaced
  by a half-swap permutation (mean TV 0.65 for spread users, 0.99 for focused users);
  (ii) sampling without repeats would exhaust a focused user's main category
  (140–180 expected draws from 50 items), so repeats are allowed (D-24).
- **2026-10-26 (M2):** the author started M2 without approving D-19 … D-21. M2 doesn't
  depend on D-19 (a baseline) or D-21 (statistics). It is consistent with draft
  hypotheses H3 and H5 (noisy scenario, `κ` factor), but they stay pending.

- **2026-10-26 (M3):** observation, not a defect: a high **repeat rate** (61% of a
  focused user's stable-scenario clicks repeat an earlier item). It follows from the
  approved choices: repeats allowed (D-24), log-normal popularity with `s_pop = 1`,
  `γ = 1`. Consequence for later milestones: a popularity-within-category Top-K list
  will match many held-out next items, so next-item metrics may reward popularity
  strongly. To revisit at M5/M7 (e.g. report popularity-only results, and consider
  `s_pop` sensitivity). No change made without approval.
- **2026-10-26 (M3):** the parent folder `IRE Project/` held a stale checkout after
  the author made `ire-adaptive-dpp-rec/` the main folder. Its `.git` was checked
  (all commits, up to `391744c`, are contained in the main repo and on GitHub; no
  stash) and deleted on the author's instruction.

- **2026-11-01 (M6 preparation):** Steck (2018) could not be read: ACM returns 403 and no
  open copy was found. Only a survey's description is available. D-19 requires the full
  text, so the calibration baseline is blocked; the author may be able to supply the PDF.
- **2026-11-01 (M6 preparation):** a throwaway prototype showed Chen et al.'s linear
  trade-off is unsuitable for our probability-scale scores (coverage 0.369 → 0.287 at
  `θ = 0.95`). Log-score form proposed (D-14).
- **2026-11-01 (M6 preparation):** correction: the literature matrix said Ibrahim et al.
  have a trade-off parameter; arXiv v1 does not. Recorded in the reading notes.

## 16. Failed approaches and negative results
*No implemented approach has failed yet.* Design-stage alternatives that were
rejected on evidence are recorded in §15 (M2: top-2 swap shift; no-repeat
sampling).

**2026-11-02 (M8), negative results on evaluation seeds** (details in §14):
- Diversification did not improve relevance for the adaptive profile: Adaptive DPP has lower
  post-shift NDCG than Adaptive Top-K (0.1762 vs 0.1794). The interaction is substitution, not
  complementarity.
- The proposed configuration is not the best adaptive setting tested for post-shift NDCG: the
  uniform-start ablation (`α = 0.07`) is about 6% higher.
- Adaptive DPP's EU loss was 3.04%, slightly above the 3% limit met on tuning seeds.
- Adaptation has real costs: worse stable tracking (H2), more false switches (H3), and about 1.8%
  lower NDCG with stable interests (H4b, inside the ±3% margin).

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

Added in Milestone 2:
11. **Repeats allowed (D-24).** Recommending already-consumed items is allowed and
    can be rewarded. Results apply to domains with repeat consumption.
    [State this as a limitation; it is consistent across all methods.]
12. **Fixed switch times.** `τ` is the same for every user. An estimator can't
    exploit this (none sees time-of-switch information), but tuning `α` on a pooled
    mix could implicitly favour settings that suit these particular switch times.
    [Sensitivity check with a different `τ` in M9, if time allows.]
13. **Common random numbers.** Scenarios share random draws, so they are *not*
    independent samples. [Analyse scenario comparisons as paired; never pool them as
    independent.]
14. **Uniform noise model.** Real off-interest clicks may be popularity-driven.
    [Stated limitation.]

Added in Milestone 7:
23. **Selected `θ` sits near the rule's limit** (2.8% vs 3%). [Report the full `θ` curve.]
24. **The rehearsal shows tuning-seed effect directions.** Since the freeze followed the
    rehearsal, state plainly that the plan was fixed by the D-30 approval *before* it ran, and
    that nothing changed afterwards except recording the selected `θ`.

Added in Milestone 6:
20. **Adapted objective.** Our log-score form differs from Chen et al.'s linear form, so
    their published results don't transfer directly. [State this in the report.]
21. **Greedy quality measured only on tiny instances.** [Report it that way.]
22. **Coverage is provisional and repeat-heavy** (risk 19): "seen" stays 0.5–0.9 for all
    profiles, including oracle-z. [Metrics decided at M7; `θ` chosen only after that.]

Added in Milestone 5:
19. **Lists dominated by already-clicked items.** In the M5 check, 69–87% of adaptive Top-K
    list items had already been clicked. This strengthens risk 15. Every method benefits
    from repeats, but a DPP's diversity may look costly partly because it moves away from
    these repeated popular items. [Decide at M7 whether to add a "new items only"
    target variant; report the seen fraction for all methods.]

Added in Milestone 4:
16. **EWMA initialisation confound (D-27).** With the uniform start, the stable-scenario
    gap between EWMA and the cumulative mean mixes two causes: forgetting and the
    prior. [Pending D-27; recommendation: bias-corrected EWMA.]
17. **Tuning objective depends on the scenario mix.** `α*` minimises error over a
    pooled mix in which 2 of 5 scenarios have no drift. A different mix would change
    `α*` (the oracle α ranges 0.02–0.15). [Report the oracle-tuned sensitivity (D-25).
    The mix is fixed before M8.]
18. **Tuning for estimation vs for recommendation.** `α*` here minimises estimation
    error. The recommender's best `α` (M6/M8) may differ. [Decide at M6 whether the
    recommenders reuse `α*` or retune on a recommendation metric, using tuning seeds
    only.]

Added in Milestone 3:
15. **Popularity-dominated repeats.** With repeats allowed and skewed within-category
    popularity, many next items are already-seen popular items (§14). Relevance
    metrics could then favour popularity-heavy rankers regardless of adaptation.
    [Report a popularity-only baseline; decide at M5/M7 whether a no-repeat target
    variant or an `s_pop` sensitivity check is needed.]

## 18. Decisions pending approval
As of 2026-11-01 (after M7): **the scale part of D-18 (proposal written, needed for M8)**;
D-07 (deadline, since M0). D-19's
calibration baseline is blocked until Steck (2018) is available.

As of 2026-11-02 (after M8): D-18 is approved. Open: D-07 (deadline); D-19 is still blocked; the
scope of M9 (which sensitivity analyses, and on which seeds, because the evaluation seeds are spent).

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
- **2026-10-26:** the author approved D-02 … D-06 as recommended. pytest 9.1.1 was
  installed. The deadline (D-07) has not been given yet.
- **Status:** completed. Milestone 1 was started on the author's instruction while
  the deadline was still missing.

### Milestone 1 — Literature review and hypothesis specification (2026-10-26)
- Ran targeted web searches. 30 sources were checked at metadata level (M) against
  publisher, DBLP/Webis anthology, proceedings or arXiv records, and recorded in
  `literature/literature_matrix.md` with method, data, metrics, stated findings and
  relevance. No source was read in full (level F); the reading list for M6 is in the
  matrix.
- Corrections found during verification: Ding & Li (2005) is by **Yi Ding and Xue
  Li**. The Wang et al. survey has six authors (including Defu Lian) and is CSUR
  54(7), Art. 154. Chen et al.'s third author appears as "Eric Zhou" (NeurIPS) and
  "Hanning Zhou" (arXiv).
- Gap analysis: **no novelty claim for the method.** Interest-weighted or
  personalised DPPs (Ibrahim et al. 2025; Liu et al. 2019 D²RL), personalised
  diversification (Teo et al. 2016) and time-decayed profiles (Ding & Li 2005; Koren
  2009) all exist. The defensible contribution is a *controlled, factorial
  analysis* with ground-truth interests across drift types. We did not find such a
  study, which is not proof that none exists.
- Proposed hypotheses H1–H6 (§20) and new decisions D-19 (calibration baseline),
  D-20 (hypotheses) and D-21 (statistical protocol).
- No code, tests or experiments.
- **Status:** D-19 … D-21 not yet approved. The author started M2 on 2026-10-26.

### Milestone 2 — Data representation and simulation design (2026-10-26)
- Wrote the simulator specification `SIMULATION_DESIGN.md`: catalogue (categories,
  embeddings, popularity), latent interest (Dirichlet concentration `κ`, half-swap
  shift), five scenarios, the interaction model (noise, popularity, hidden taste `λ`),
  the observable/prohibited information table, the seed protocol with common random
  numbers, scale, and limitations.
- Ran throwaway Monte Carlo checks (scratchpad, not in the repo) to choose `σ`,
  `κ` levels and the shift construction (§15). These are design evidence, not
  experimental results.
- Updated D-08 … D-10 and added D-22 … D-26.
- No project code, no tests, no experiments. **No git commit** (per the author's
  instruction of 2026-10-26: never commit without explicit permission).
- **2026-10-26:** the author approved the M2 design (D-08, D-09, D-10, D-22 … D-26,
  seed protocol) and the M1 decisions D-19 (option b), D-20 and D-21, all as proposed.
  `hypothesis.md` was created from the approved H1–H6. Not committed.
- **Update:** the M2 changes were committed outside this session as `391744c`
  ("MIlestone-2", author `Salvotore-Originals`) and pushed to origin. On 2026-10-26
  the author made the clone `IRE Project/ire-adaptive-dpp-rec/` the main folder.
- **Status:** complete.

### Milestone 3 — Simulator implementation and tests (2026-10-26)
- Implemented `adaptdiv/` (`config`, `seeding`, `catalogue`, `scenarios`,
  `simulator`) exactly as specified in `SIMULATION_DESIGN.md` (§6 of this log).
- Wrote 39 pytest tests (§13). All pass.
- Added `experiments/sim_check.py`. Ran it on tuning seeds 0–4 (§14): scenario
  behaviour as designed, generation about 0.23 s per seed.
- New observation: high repeat rate driven by popularity (§15, risk 15).
- Cleaned up the parent folder on instruction (§15).
- No git commit (the author's standing instruction).
- **Status:** reviewed. Committed as `ea1cc38` and pushed on the author's
  instruction.

### Milestone 4 — Adaptive interest estimation (2026-10-28)
- D-11 approved by the author. Implemented `adaptdiv/estimators.py` (cumulative mean,
  EWMA, sliding window, matched-memory helpers) and the provisional
  `adaptdiv/estimation_metrics.py` (tracking error, crossover lag, false-switch rate,
  noise floor).
- 26 new tests: hand-checked values, unrolled-formula equivalence, limiting cases,
  simplex invariants, no look-ahead, batch = single, convergence, effect of an
  isolated noisy click, metric hand examples. Full suite: **65 passed**.
- `experiments/estimator_study.py` on tuning seeds 0–4: `α* = 0.07`, `W* = 40` (pooled
  objective). Per-scenario results in §14.
- Found the EWMA initialisation confound and raised D-27 (not changed without
  approval).
- Moved the assistant's project memory to the new folder's memory path (no repo
  change).
- No git commit.
- **Update:** the M4 changes were committed outside this session as `c4b9cfc`
  ("Milestone-4", author `Salvotore-Originals`) and pushed.
- **2026-11-01:** added `ewma_bias_corrected` as an extra option (default unchanged), 6
  tests (71 total), and `experiments/ewma_init_comparison.py` with its run (§14) as
  evidence for D-27. Noted that D-13's preliminary "exclude consumed items" conflicts with
  approved D-24. Not committed.
- **2026-11-01:** the author approved D-27 option (b). Added `PROPOSED_ESTIMATOR`,
  `STATIC_ESTIMATOR` and `UNIFORM_START_ABLATION` to `estimators.py`, and an amendment to
  `hypothesis.md` (before any evaluation run). The window matched to `α = 0.05` is
  `W = 39`, almost the same as the tuned `W* = 40`.
- **2026-11-01, design evidence for M5** (throwaway script, tuning seeds 0–4): category
  totals of global `r_i` range about 0.09–0.11 (max/min ≈ 1.2); within-category estimates
  correlate 0.996 with true `π_i`. See the D-13 proposal.
- **Status:** complete, not committed.

### Milestone 5 — Candidate generation and base relevance (2026-11-01)
- D-12 and D-13 approved by the author. Implemented `adaptdiv/relevance.py` and
  `adaptdiv/similarity.py`.
- 13 new tests; full suite **84 passed**.
- `experiments/relevance_check.py` on tuning seeds 0–4 (§14): scores behave sensibly and
  most listed items were already clicked (risk 19).
- Not committed.
- **Update:** committed as `c3e2417` and pushed on the author's instruction.
- **Status:** complete.

### Milestone 6 preparation (2026-11-01; no project code)
- Read the relevant sections of Kulesza & Taskar, Chen et al., Wilhelm et al., Ibrahim et
  al. and Lin et al. (`literature/literature_matrix.md`, reading notes). Steck (2018) was
  inaccessible.
- Throwaway prototype on tuning seed 1 (scratchpad, not in the repo): log-score trade-off
  behaves smoothly, the linear form doesn't; about 0.2–0.4 ms per list.
- Rough M8 cost from that speed: 4M lists per method, so about 20 minutes per DPP method
  on one core, less when run in parallel over seeds. To be measured in M6.
- Proposals written for D-14, D-15, D-16; D-28 added. Not committed.
- **Status:** complete. D-14, D-15, D-16 and D-28 approved by the author.

### Milestone 6 — DPP and recommendation baselines (2026-11-01)
- Implemented `adaptdiv/dpp.py` and `adaptdiv/recommenders.py` (§6). 44 new tests; full suite
  **128 passed**.
- `experiments/dpp_check.py` on tuning seeds (§14): greedy quality, speed, `θ` sweep. No `θ`
  selected yet.
- The calibration re-ranker (D-19) was **not** implemented: Steck's paper is unavailable.
- The dataviz palette validator could not run (Node.js isn't installed); plot series are also
  separated by line style and order.
- Rough M8 cost at 0.15 ms per list: 4M lists is about 10 minutes per DPP method on one core.
- Not committed.
- **Status:** complete.

### Milestone 7 preparation (2026-11-01; no project code)
- Throwaway probe of candidate metrics on tuning seeds (scratchpad, not in the repo). Five-step
  targets are far less noisy than one next item; sorting DPP lists by score removes a small
  ordering penalty (0–0.9%); repeats are 33–64% of targets, and the DPP's change on new items
  (+0.9% to −4.3%) is about the same as on all items, so risk 19 is smaller than feared.
- Wrote the D-17 proposal with exact formulas, D-29 (`θ` rule) and D-30 (8 confirmatory tests
  under Holm, plus H4b and H6a checks). Not committed.
- **Status:** complete. D-17, D-29 and D-30 approved by the author.

### Milestone 7 — Evaluation framework (2026-11-01)
- Implemented `metrics.py`, `evaluation.py`, `stats.py`, `confirmatory.py`; 31 new tests; full suite
  **159 passed**.
- `experiments/select_theta.py` selected `θ = 0.5` on tuning seeds and rehearsed the pipeline (§14).
- Froze the analysis plan in `hypothesis.md`. No evaluation seed has been run.
- Not committed.
- **Status:** awaiting review. Milestone 8 needs the D-18 scale decision. Recommended: commit the
  freeze before any evaluation run, so git records that the plan came first.

### Milestone 8 — Main experiments and ablations (2026-11-02)
- Added `experiments/main_experiment.py`. It refuses a full run if `adaptdiv/` differs from HEAD,
  and a rehearsal needs its own output folder. No `adaptdiv/` file or test changed; 159 tests pass.
- Rehearsed on tuning seeds, then ran evaluation seeds 100–119 once (§14).
- The script itself was uncommitted when it ran; `environment.json` records its SHA-256.
- Calibration not run (D-19 blocked).
- **Status:** awaiting review. Results not committed.

## 20. Research questions ↔ hypotheses ↔ experiments ↔ conclusions

### Hypotheses (M1, 2026-10-26) — APPROVED 2026-10-26 (D-20); canonical copy in `hypothesis.md`
These are **predictions, not results**. Each states what would *disconfirm* it.
Metric formulas are fixed at M7, and the hypotheses are frozen before any
evaluation-seed run. Notation: TV(a,b) = ½‖a−b‖₁ (total-variation distance);
τ = shift time; "post-shift window" = the interactions τ … τ+Δ (Δ to be fixed at M7).
"Drift scenarios" = abrupt, gradual and recurrent. All comparisons are paired
across evaluation seeds under the D-21 protocol.

| ID | RQ | Hypothesis (directional prediction) | Disconfirmed if | Expected to favour proposed method? |
|---|---|---|---|---|
| **H1** Responsiveness | RQ1 | In drift scenarios, EWMA (α tuned on tuning seeds) has (a) a shorter adaptation lag and (b) lower mean TV(z_t, ẑ_t) over the post-shift window than the cumulative mean | The paired difference is not significant, or significantly favours the cumulative mean | Yes |
| **H2** Cost of adaptation | RQ1 | In the stable scenario, after burn-in, the cumulative mean has lower mean TV error than EWMA (forgetting adds variance when nothing changes) | EWMA is equal or better | **No.** It predicts a loss for the adaptive estimator and guards against reporting only favourable scenarios |
| **H3** Noise sensitivity | RQ1 | In the noisy scenario, EWMA's estimated dominant category (argmax ẑ_t) switches more often while the true dominant category is unchanged ("false switches") than the cumulative mean's does | Equal or fewer false switches | **No** |
| **H4** Adaptation → relevance | RQ1/RQ3 | In drift scenarios, Adaptive Top-K has higher post-shift NDCG@K (and expected utility) than Static Top-K. In the stable scenario the difference is below the SESOI | No significant post-shift gain; or a significant stable-scenario difference beyond the SESOI | Yes (drift) / neutral (stable) |
| **H5** Diversification trade-off | RQ2 | With the same profile, DPP gives higher category coverage than Top-K, and its NDCG@K cost relative to Top-K is smaller for spread interests (high-entropy z) than for focused ones (low entropy): a selector × concentration interaction | No coverage gain, or no (or reversed) interaction with concentration | Partly |
| **H6** Combined method | RQ3 | (a) In drift scenarios, Adaptive DPP is not Pareto-dominated in (NDCG@K, category coverage) by any compared method. (b) The adaptation × selector interaction on post-shift NDCG@K is tested **two-sided**, because there are two plausible rival predictions: *complementarity* (adaptation helps the DPP more) vs *substitution* (the DPP's hedging already absorbs some of the drift, so adaptation helps it less) | (a) Another method dominates it on both axes. (b) Reported in whichever direction it falls; neither sign confirms "superiority" | Undetermined (b) |

Exploratory (no directional prediction, not in the confirmatory family): EWMA vs
sliding window at matched memory; sensitivity to α, W, η, κ, drift speed and the DPP
trade-off parameter; the calibration re-ranker (if D-19 is approved) vs Adaptive DPP.

| RQ | Hypotheses | Experiment (planned) | Conclusion |
|---|---|---|---|
| RQ1 | H1, H2, H3, H4 | M4 estimator study; M8 main runs | — (not run) |
| RQ2 | H5 | M6 small examples; M8 main runs | — (not run) |
| RQ3 | H4, H6 | M8 2×2 factorial runs | — (not run) |

Update 2026-11-02 (M8, this simulation only; full conclusions wait for M10):
| RQ | Measured finding |
|---|---|
| RQ1 | EWMA adapts much faster and gains large post-shift relevance; it pays in stable tracking and false switches (H1–H4 supported) |
| RQ2 | DPP adds coverage at about 1.8% NDCG cost; the focused/spread difference is very small (H5 supported) |
| RQ3 | Substitution: the DPP helps a static profile slightly and costs an adaptive one slightly. Adaptive DPP is not dominated (H6) |

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

### Milestone 1
- **Is your method novel?** No. Interest-weighted DPP quality (Ibrahim et al. 2025),
  DPP kernels that adapt over interactions (Liu et al. 2019) and time-decayed
  profiles (Ding & Li 2005; Koren 2009) already exist. The contribution is a
  controlled study that uses ground-truth interests to separate the effects of
  adaptation and diversification, and their interaction, across drift types.
- **Why include hypotheses you expect to lose (H2, H3)?** Adaptation trades
  responsiveness for stability. A study that only reports drift scenarios would
  overstate the benefit. Predicting where the method should lose makes the
  evaluation falsifiable.
- **Why is H6(b) two-sided?** Diversification may *substitute* for adaptation: a
  diverse static list already covers the new interest. It may also *complement* it.
  Both are plausible, so we don't pre-commit to a direction.
- **How is calibration different from a DPP?** Calibration (Steck 2018) matches the
  list's category *proportions* to the profile. A DPP trades off item quality against
  pairwise similarity in embedding space, with no explicit target proportions.
- **Why can't greedy DPP be called "the MAP"?** Selecting the max-determinant subset
  is NP-hard (Ko, Lee & Queyranne 1995). Greedy is a fast heuristic, and better
  approximations exist (Gillenwater et al. 2012).

### Milestone 3
- **How do you guarantee reproducibility?** Every random draw comes from
  `SeedSequence(entropy=[PROJECT_TAG, seed], spawn_key=path)`. The path names *what*
  is drawn (catalogue, training user `j`, evaluation user `(k, j)`, params vs
  interactions), so results depend only on (seed, path). Tests check that the same seed
  gives identical data and that adding users leaves existing users unchanged.
- **What is inverse-CDF sampling, and why use it here?** To draw from probabilities
  `p`, take `U ~ Uniform(0,1)` and return the first index whose cumulative sum
  exceeds `U`. Because we store `U` per user and step, every scenario reuses the same
  `U`. Two scenarios with the same `z_t` at step `t` therefore produce the same click.
  That is how "identical before the switch" is achieved and tested.
- **How do you know the simulator follows `z_t`?** A 200,000-step test shows
  empirical category frequencies within TV 0.01 of `(1−η) z + η/C`. In the check
  run, windowed tracking error is flat over time, so there is no generator lag. A test
  also checks that post-switch clicks match `z_B`, not `z_A`.
- **How is leakage prevented in code?** `ObservedSequence` has exactly one field
  (`items`), checked by a test. Hidden quantities live only in `GroundTruth`, and all
  arrays are read-only. Later milestones' recommenders will receive
  `ObservedSequence` + `PublicCatalogue` + training popularity only.
- **Why is the repeat rate so high, and does it matter?** Popularity is skewed (the
  top-5 items take 40% of a category's clicks) and repeats are allowed, so focused
  users re-click popular items. It matters for next-item metrics, which could reward
  popularity. We will report a popularity baseline and revisit it at M5/M7.

### Milestone 7
- **How do you stop results influencing the analysis?** Metrics, tests, thresholds, `α` and `θ` were
  fixed and written into `hypothesis.md`, using tuning seeds only, before any evaluation seed ran.
- **Why five held-out clicks?** One click is very noisy. Five reduced the paired noise from up to
  17% to about 2%.
- **What does "supported" mean here?** Holm-adjusted p < 0.05 with the predicted sign. Practical
  importance is a separate check: the whole 95% CI must lie beyond the SESOI.
- **What does Holm do?** Testing eight hypotheses raises the chance of a false positive. Holm compares
  the smallest p with `α/8`, the next with `α/7`, and so on, controlling the family-wise error.
- **Why was `θ = 0.5` chosen?** It is the most diverse setting losing at most 3% expected utility for
  both profiles. Maximising relevance alone would just pick Top-K.
- **Does sorting a DPP list change the set?** No, only the order. It gives both selectors the same
  ranking rule.

### Milestone 6
- **Is your objective a real DPP?** Yes, for `θ < 1`: with `q_i = s_i^{θ/(2(1−θ))}`,
  `F(Y) = (1−θ) log det L_Y`, and `L` is PSD because `S` is PSD and `q` is positive. A test
  checks this identity.
- **Why not Chen's objective as published?** It adds raw scores to a log-determinant. Our
  scores are small probabilities, so lists collapsed into ignoring relevance. Logs are
  scale-free, and `θ = 1` is exactly Top-K.
- **Walk through one greedy step.** `d_i²` is how much adding item `i` multiplies `det S_Y`.
  Pick the best `θ log s_i + (1−θ) log d_i²`, then update every `d_i²` by removing the part
  of `i` explained by the newly picked item.
- **Is greedy optimal?** No. MAP is NP-hard, and greedy's known guarantee is weak. On tiny
  instances it matched the exact optimum 95–100% of the time, but that doesn't bound it at
  `N = 500`.
- **Why does diversity help the static profile after a shift?** A stale profile puts a Top-K
  list entirely in the old categories. A diverse list spreads out, so some items land in
  the new interests. That is hedging, and it may partly substitute for adaptation (H6b).

### Milestone 5
- **Why within-category popularity?** Global popularity also records how often each
  category happened to be clicked in training, which is noise here (category totals
  0.09–0.11 for identical categories). Within-category popularity leaves category
  preference entirely to `ẑ_t`.
- **What does the plug-in score mean?** `r_i ≈ P(i | c_i)` and `ẑ_{t,c} ≈ P(c)`, so
  `r_i · ẑ_{t,c_i}` estimates the next-item probability, and it sums to 1 over items.
- **Why is `S` guaranteed PSD?** `vᵀΦΦᵀv = ‖Φᵀv‖² ≥ 0` for every `v`. The jitter adds
  `ε` to every eigenvalue.
- **Does the jitter solve the rank limit?** No. `ΦΦᵀ` has rank at most `d`, and beyond
  `d` items only the tiny `ε` remains. That is why we need `d ≥ K`.
- **Why are no items excluded?** Repeat clicks exist in the simulator (D-24), so excluding
  seen items would make those targets impossible to hit.

### Milestone 4
- **Why does EWMA stay on the simplex without renormalising?** If `ẑ_{t−1}` is a
  probability vector and `x_t` is one-hot, then `(1−α) ẑ_{t−1} + α x_t` is a convex
  combination of two probability vectors. It is non-negative and sums to
  `(1−α) + α = 1`.
- **What does α control?** The weight of an observation of age `a` is `α(1−α)^a`. Large
  `α` forgets fast: short lag, but noisy and prone to false switches. Small `α`
  averages more: stable, but slow after drift. In our tuning-seed study, lag fell from
  about 30 to under 1 step as `α` went 0.02 → 0.5, while noisy false switches rose
  from 2% to 68%.
- **What happens after one noisy click?** EWMA moves exactly `α` of the way towards
  that click's category (tested). With `α = 0.07` one off-interest click can't change
  the dominant category of a focused user.
- **Why compare at matched memory, and what did we find?** `W = 2/α − 1` gives EWMA and
  the window the same mean observation age. Even so, EWMA crossed over faster (9.3 vs
  13.5 steps), because geometric weights favour the newest observations.
- **Why is the cumulative-mean lag "censored"?** After a switch at step 100, the
  cumulative mean needs as many post-switch clicks as pre-switch ones before it leans
  towards the new interest. Within 200 steps it often never does, so the lag is only a
  lower bound.
- **What is the noise floor?** Even a perfect tracker of the click distribution is off
  by `η · TV(z, uniform)` from the true interest, because noisy clicks are spread
  uniformly. Errors near the floor can't be improved by any estimator.
- **What is the initialisation issue (D-27)?** The uniform `ẑ_0` decays as `(1−α)^t`. It
  biases EWMA towards uniform early on: bad for focused users, good for spread ones.
  The bias-corrected version uses observations only.

### Milestone 2
- **Why is `z_B` a permutation of `z_A` instead of a fresh Dirichlet draw?** A fresh
  draw would change how concentrated the user is at the same time as what they like,
  confounding drift with concentration. A permutation changes only *which* categories
  are preferred.
- **Why is noise applied to interactions and not to `z_t`?** `z_t` is the user's true
  intent, the quantity we evaluate the estimator against. Noise models clicks that
  don't reflect intent, so it belongs in the observation model. That is also why even
  a perfect tracker has a noise-floor error of `η · TV(z, uniform)`.
- **Why `λ > 0`?** With `λ = 0`, the oracle recommender's model would be exactly the
  generator. A hidden item-level taste ensures no method is correct by construction.
- **What are common random numbers, and why use them?** All scenarios reuse the same
  per-user uniform draws (inverse-CDF sampling). Differences between scenarios then
  come from the trajectory alone, not from sampling luck, which makes paired
  comparisons much less noisy.
- **Why tune `α` on a pooled mix of scenarios?** Choosing `α` per scenario would use
  the scenario label, which a real system wouldn't know. That is indirect leakage.

### Milestone 8
- **Does this prove the method is better?** No. It is one synthetic setup. Adaptive DPP offers a
  different trade-off from Adaptive Top-K: more coverage for slightly lower relevance.
- **Why is every p-value 1.91e-06?** All 20 seeds had the same sign, which is the smallest p a
  two-sided Wilcoxon can give with 20 pairs. So the p-value says nothing about size; use the CI.
- **What does substitution mean here?** The DPP's spread hedges against a stale static profile, so it
  helps there. An adaptive profile is already current, so the same spread only costs relevance.
- **Why not switch to `α = 0.07` after seeing the ablation?** That would tune on evaluation seeds. The
  ablation also changes the start as well as `α`, so it can't identify the cause.
- **H4b passed, so is adaptation free on stable interests?** No. It is about 1.8% lower, with a CI that
  excludes zero. "Equivalent" only means inside the ±3% margin we set in advance.
