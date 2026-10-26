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
Still pending: D-07 (deadline); D-11 … D-17 and the scale part of D-18 (confirmed
at their own milestones). Scope from the master prompt:
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
│   ├── estimators.py       cumulative mean, EWMA, sliding window
│   ├── relevance.py        base relevance r_i, quality transform g
│   ├── similarity.py       S = ΦΦᵀ (+ jitter), PSD checks
│   ├── dpp.py              greedy MAP (incremental Cholesky) + naive reference
│   ├── recommenders.py     Top-K / DPP × static / adaptive / oracle
│   └── metrics.py          estimation, relevance, diversity, responsiveness
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

## 9. Baselines and ablation studies [PLANNED — D-16]
A 2×2 design of {static, adaptive} × {Top-K, DPP}, plus popularity, plus an oracle-z
diagnostic ceiling. Estimator variants: EWMA vs sliding window. Sensitivities: `α`/`W`,
noise `η`, interest concentration `κ`, drift speed `W`, and the DPP trade-off.

## 10. Evaluation metrics and calculation procedures [PLANNED — D-17]
Exact formulas will be written here at M7 and hand-checked in tests.

## 11. Commands used to run experiments
Run from the repository root (`ire-adaptive-dpp-rec/`):
```bash
python -m pytest -q                      # full test suite
python experiments/sim_check.py          # M3 simulator check on tuning seeds 0-4
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

### Milestone 3 (2026-10-26): `python -m pytest -q` → **39 passed in 0.66 s** (pytest 9.1.1)
| File | What is checked |
|---|---|
| `test_catalogue_and_seeding.py` (8) | shapes and grouping; unit norms; popularity positive and summing to 1 per category; within/across cosine ≈ 0.50/0.00 (10 catalogues, tolerance 0.03); public view hides popularity and centroids; read-only arrays; same seed gives the same catalogue and different seeds differ; streams depend only on (seed, path); JSON config round-trip |
| `test_scenarios.py` (13) | `z_A` on the simplex (both `κ`); E[max z] ≈ 0.665/0.293 (5,000 draws, tol 0.02); half-swap is a permutation that moves every top category into the bottom half and keeps the sorted values; mean shift TV ≈ 0.99/0.65 (3,000 draws, tol 0.03); partial-shift magnitude; every scenario on the simplex; exact switch points (abrupt 100, recurrent 70/140, gradual 80–120 with `w=0.5` at t=100); noise rates |
| `test_simulator.py` (18) | conditional rows are distributions; `λ=0` reduces to `π^γ`; **200,000-step stable sequence**: noise rate within 0.005 of `η`, category-frequency TV < 0.01, item-frequency TV < 0.03 vs the model; exact marginal `P_t(i)`; item ids consistent with categories; **common random numbers** (stable = abrupt for t<100, = recurrent for t<70, = gradual for t≤80; noisy differs only where its noise flag differs); post-switch interactions follow `z_B` (pooled TV gap > 0.2); reproducible `generate`; adding users leaves existing users unchanged; different seeds differ; params shared across scenarios; **firewall** (`ObservedSequence` has only `items`, integer and read-only; ground-truth arrays read-only); training counts sum correctly and are independent of the number of evaluation users |

Statistical tests use fixed seeds, so they are deterministic. Their tolerances were set
before running, from the expected Monte Carlo error.

## 14. Experiment results
*No recommendation experiments have been run.*

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

## 16. Failed approaches and negative results
*No implemented approach has failed yet.* Design-stage alternatives that were
rejected on evidence are recorded in §15 (M2: top-2 swap shift; no-repeat
sampling).

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

Added in Milestone 3:
15. **Popularity-dominated repeats.** With repeats allowed and skewed within-category
    popularity, many next items are already-seen popular items (§14). Relevance
    metrics could then favour popularity-heavy rankers regardless of adaptation.
    [Report a popularity-only baseline; decide at M5/M7 whether a no-repeat target
    variant or an `s_pop` sensitivity check is needed.]

## 18. Decisions pending approval
As of 2026-10-26 (after M3): open items are D-07 (deadline, since M0); D-11
(estimators, needed for M4); D-12, D-13 (M5); D-14 … D-16 (M6); D-17 (M7); the scale
part of D-18 (M8).

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
- **Status:** awaiting the author's review. Milestone 4 (estimators) needs D-11.

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
