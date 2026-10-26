# SIMULATION_DESIGN.md — Catalogue, latent interests, drift, interaction model

**Status: APPROVED 2026-10-26 (Milestone 2). IMPLEMENTED and tested in Milestone 3 (`adaptdiv/`, 39 tests).**
This file is the specification that Milestone 3 implements and tests against. Any later change must be recorded as a dated
revision at the bottom; the old text is kept.

Decision IDs refer to `DECISIONS.md`. Numbers marked "MC check" come from a
throwaway Monte Carlo script run on 2026-10-26 (§9). They are design evidence, not
experimental results.

---

## 1. Overview: what is generated, in order

For one **seed** `s`:

1. **Catalogue** (shared by every user and scenario under seed `s`): categories,
   item embeddings, and item popularity.
2. **Training population** (used only to estimate base relevance `r_i`, D-13): a separate set of
   users whose interactions produce item counts.
3. **Evaluation users** `j = 1..U`. Each user gets **user parameters** that are shared across all
   scenarios: a base interest `z_A`, a shifted interest `z_B`, a hidden taste `u`, and
   a fixed bank of random numbers.
4. **Per scenario:** an interest trajectory `z_1..z_T`, then the interaction sequence
   `i_1..i_T`, generated from the same random-number bank so that scenarios are
   directly comparable (§7).

The simulation is **open-loop**: recommendations never influence the sequence (D-04).

## 2. Catalogue

| Symbol | Meaning | Proposed default |
|---|---|---|
| `C` | number of categories | 10 |
| `n_c` | items per category (equal sizes) | 50, so `N = 500` |
| `d` | embedding dimension | 32 (must be ≥ `K`, see D-12) |
| `σ` | within-category spread of embeddings | 1.0 |
| `s_pop` | log-normal popularity spread | 1.0 |

**Category centroids.** `μ_c = normalise(h_c)`, with `h_c ~ N(0, I_d)` drawn
independently, so different categories are roughly orthogonal (D-22 discusses an
optional grouped structure).

**Item embeddings.** Item `i` belongs to exactly one category `c_i` and has
`φ_i = normalise(μ_{c_i} + σ · ε_i)`, with `ε_i ~ N(0, I_d / d)`, so `‖ε_i‖ ≈ 1`.
MC check: with `σ = 1`, the mean cosine similarity is **0.50 within a category** and
**0.00 across categories** (`σ = 0.5` gives 0.80; `σ = 1.5` gives 0.31). With
`σ = 1`, items in the same category are clearly related but not duplicates, so a
DPP has something to choose between inside a category.

**Popularity.** `π_i ∝ exp(s_pop · ζ_i)` with `ζ_i ~ N(0, 1)`, normalised **within each
category**. Popularity is independent of the embedding.

**The catalogue is regenerated per seed**, so conclusions don't depend on one random
catalogue.

## 3. Latent interest `z_t`

`z_t ∈ Δ^{C−1}`, a probability vector over categories (D-08).

**Base interest.** `z_A ~ Dirichlet(κ · 1_C)`. **`κ` sets how concentrated a user's
interests are** and is a controlled factor (needed for H5):

| `κ` | Label | E[max component] | E[normalised entropy] | E[# categories holding 80% of mass] |
|---|---|---|---|---|
| 0.1 | **focused** | 0.67 | 0.37 | 2.0 |
| 0.3 | — | 0.46 | 0.62 | 3.2 |
| 1.0 | **spread** | 0.29 | 0.84 | 5.2 |

(MC check, 20,000 draws.) Proposal: two levels, `κ ∈ {0.1, 1.0}`. `κ = 0.3` is
optional as a middle level for sensitivity analysis.

**Shifted interest `z_B` (for drift).** `z_B` is a permutation of `z_A`, so the user's
interest stays *as concentrated* after the shift and only *which* categories they
like changes. This keeps concentration from being confounded with drift. The
permutation is a **half-swap**: the top-`C/2` categories under `z_A` are mapped by a
random bijection onto the bottom-`C/2` categories, and vice versa.

| Shift construction | `κ=0.1` mean TV (5th pct) | `κ=1.0` mean TV (5th pct) |
|---|---|---|
| swap only the top-2 categories | 0.84 (0.54) | **0.36 (0.16)**, too weak for spread users |
| **half-swap (proposed)** | 0.99 (0.95) | 0.65 (0.45) |

(MC check, 5,000 draws; TV = ½‖z_A − z_B‖₁.) An optional *shift magnitude* `m ∈ [0,1]`,
with `z_B(m) = (1−m) z_A + m · z_B`, allows partial shifts in sensitivity analysis.
The default is `m = 1`.

## 4. Drift scenarios (trajectories)

`T = 200` interactions per user. All switch points are fixed so that post-shift
windows line up across users.

| Scenario | Definition of `z_t` | Defaults | Tests |
|---|---|---|---|
| **Stable** | `z_t = z_A` | — | H2 (cost of adaptation), H4 stable arm |
| **Abrupt** | `z_A` for `t < τ`; `z_B` for `t ≥ τ` | `τ = 100` | H1, H4, H6 |
| **Gradual** | `(1−w_t) z_A + w_t z_B`, `w_t = clip((t − τ₀)/W, 0, 1)` | `W = 40`, `τ₀ = 80` (centred on 100) | H1, H4, H6 |
| **Recurrent** | `z_A` for `t < τ₁`; `z_B` for `τ₁ ≤ t < τ₂`; `z_A` for `t ≥ τ₂` | `τ₁ = 70`, `τ₂ = 140` | H1, H6; whether old interests are "remembered" |
| **Noisy** | Stable `z_A`, with a high interaction-noise rate `η = 0.3` | `η_noisy = 0.3` | H3 (false switches) |

All trajectories stay on the simplex: convex combinations and permutations of simplex
points are still simplex points. **Noise is not part of `z_t`.** It acts at the
interaction level (§5), so `z_t` remains "what the user really wants".

Note for H3: the true dominant category `argmax z_t` is constant in the noisy
scenario. Any change in `argmax ẑ_t` there is therefore a *false switch*.

## 5. Interaction-generation model (D-09)

At each step `t = 1..T`, for user `j` with interest `z_t`, taste `u`, and noise rate `η`:

1. **Noise flag:** `n_t ~ Bernoulli(η)`. Base `η = 0.05` in every scenario, and
   `0.3` in *noisy*.
2. **Category:** if `n_t = 0`, `c_t ~ Categorical(z_t)`; if `n_t = 1`,
   `c_t ~ Uniform{1..C}`.
3. **Item within the category:**
   `P(i_t = i | c_t, u) ∝ π_i^γ · exp(λ · ŝ_i(u))` for items `i` with `c_i = c_t`,
   where `ŝ_i(u)` is `⟨φ_i, u⟩` standardised over the catalogue (zero mean, unit
   variance), and `u = normalise(N(0, I_d))` is a fixed hidden taste vector for the user.
   Defaults: `γ = 1`, `λ = 0.5`.
4. **Repeats are allowed** (D-24). MC check: a focused user (top-category mass
   0.7–0.9) would draw 140–180 times from a 50-item category in `T = 200` steps, so
   sampling without replacement would exhaust the category and distort the model.

So the probability of item `i` at time `t` is
`P_t(i) = [(1−η) z_{t,c_i} + η/C] · P(i | c_i, u)`.
The simulator computes this exactly, so evaluation can use expected-utility metrics (D-17).

**Why `λ > 0` (partial misspecification).** The recommender knows categories,
embeddings and estimated popularity, but not `u`. With `λ = 0`, a recommender using
the true `z_t` would match the generator exactly (an oracle that is right by
construction). `λ = 0.5` adds item-level preference that the method cannot capture.
`λ ∈ {0, 1}` is used as a sensitivity check.

**Effect of noise on the target.** The observed category distribution is
`(1−η) z_t + η/C`, not `z_t`. Even a perfect tracker of observed categories has a
noise-floor error of `TV = η · TV(z_t, uniform)` against `z_t` (risk 6 in the log). This
is reported, not corrected.

## 6. Observable vs prohibited information (leakage firewall)

| Information | Estimator / recommender | Evaluation (metrics) |
|---|---|---|
| Item ids of the interactions `i_1..i_t` (up to the current step) | **allowed** | allowed |
| Item catalogue metadata: `c_i`, `φ_i` | **allowed** (public item features) | allowed |
| Popularity *estimated* from the training population (`r_i`, D-26) | **allowed** | allowed |
| Future interactions `i_{t+1..}` | **prohibited** | allowed (held-out targets) |
| `z_t`, `z_A`, `z_B`, switch times, scenario label | **prohibited** | allowed |
| Noise flags `n_t`, `η` | **prohibited** | allowed |
| Taste `u`, `λ`, true popularity `π_i`, `γ`, `κ` | **prohibited** | allowed |

**Enforcement (in M3).** The simulator returns `ObservedSequence` (item ids only) and
`GroundTruth` (everything else) as **separate objects**. Recommender and estimator
functions take an `ObservedSequence` truncated at `t` plus a `PublicCatalogue` view
(categories and embeddings only). A test will check that these types carry no
ground-truth fields.

**Leakage through tuning (D-25).** If `α` (or `W`) is tuned separately for each
scenario, the method indirectly learns the scenario label. Proposal: tune **one**
setting per method over a pooled mix of scenarios on tuning seeds (realistic, no
leakage). Per-scenario tuning is reported only as an "oracle-tuned" sensitivity
result.

## 7. Random-seed protocol (D-18, refined)

- **Seeds:** tuning seeds `{0,…,4}`; evaluation seeds `{100,…,119}` (20 seeds).
  Evaluation seeds are never used for any tuning or design choice.
- **Stream separation** with `numpy.random.SeedSequence(entropy=[PROJECT_TAG, s])`.
  Children are spawned in a fixed order: `catalogue`, `training_population`,
  `users`. The `users` child spawns one child per user, which spawns
  `params` (`z_A`, half-swap permutation, `u`) and `interactions`.
- **Common random numbers across scenarios.** For each user, the `interactions`
  stream pre-draws uniform arrays `U_noise[t]`, `U_cat[t]`, `U_item[t]`, and each draw
  uses inverse-CDF sampling. All five scenarios reuse the same arrays, so:
  - before the first switch, a user's *abrupt*, *recurrent* and *stable* sequences
    are **identical**;
  - differences between scenarios come only from the trajectory, never from luck;
  - comparisons between scenarios are paired, which reduces variance.
  (The noisy scenario still uses the same `U_noise`, but with a higher threshold `η`.)
- Adding a user or a scenario must not change existing draws. Scenarios don't
  consume random numbers, and users are spawned by index.
- Every run saves the config, seed list, NumPy version and git commit hash
  (once commits exist) under `results/<run_id>/`.

## 8. Scale and cost (preliminary)

Per seed: 2 concentrations × 5 scenarios × `U = 100` users × `T = 200` = 200,000
interactions. Over 20 evaluation seeds, that is 4M interactions. Generation is
vectorised NumPy, expected to take seconds to about a minute. This will be measured
in M3, not assumed. Recommendation cost (M6–M8) dominates and is estimated separately.

Training population (D-26): `U_train = 1,000` users × 50 interactions per seed, with
stable interests drawn from the same Dirichlet mixture over both `κ` levels.
`r_i = (count_i + 1) / (Σ counts + N)` (add-one smoothing, so `r_i > 0`).

## 9. Design evidence: Monte Carlo checks (2026-10-26)

Throwaway scripts kept outside the repository (scratchpad) produced the numbers in
§2–§5:
embedding cosine by `σ`; Dirichlet concentration by `κ`; shift size for two
permutation schemes; category-exhaustion arithmetic. They can be rerun and will be
re-verified by unit tests in M3. Re-running with other RNG seeds can change the
last decimal place.

## 10. Known limitations of this design
- Single-label items and equal-size categories (no long tail of categories).
- Interests change only through scheduled switches. There is no slow random-walk
  drift (possible extension).
- Noise is uniform over categories. Real "noise" may favour popular items.
- The user taste `u` is static and only weakly modelled.
- The recommender's assumptions (category-level interests) partly match the
  generator. `λ`, `κ`, `η` and the stable scenario are the guards against a simulator
  that favours the method, but they don't remove the risk.

## Revision history
- 2026-10-26 — Created (Milestone 2 proposal).
- 2026-10-26 — Implemented in M3 with no change to the specification. Implementation detail: a single uniform per step serves both the signal and the noise category draw (see `IMPLEMENTATION_LOG.md` §6).
- 2026-10-26 — Approved by the author as proposed (D-08 … D-10, D-22 … D-26, seed protocol).
