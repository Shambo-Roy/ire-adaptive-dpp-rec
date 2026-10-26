# hypothesis.md — Research hypotheses

**Status: APPROVED 2026-10-26 (D-20).** These are predictions, not results.

**Freeze rule.** Exact metric formulas are fixed at Milestone 7. After that, the
hypotheses, metrics, SESOI and the analysis plan are **frozen before any
evaluation-seed run** (Milestone 8). Any later change must be added below as a dated
amendment with its reason, and marked as made *before* or *after* seeing evaluation
results. Earlier text is never edited away.

## Research questions
- **RQ (central):** Can adaptive interest estimation combined with interest-weighted
  DPP selection improve the balance of relevance, diversity and responsiveness to
  changing intent, compared with non-adaptive and non-DPP baselines?
- **RQ1 (adaptation):** Does an adaptive estimator track the true interest `z_t`
  better after drift, and at what cost when interests are stable or clicks are noisy?
- **RQ2 (diversification):** How much does DPP selection gain in diversity, and lose
  in relevance, for focused vs spread interests?
- **RQ3 (interaction):** Do adaptation and DPP diversification complement each other
  or substitute for each other?

## Conventions
- TV(a,b) = ½‖a − b‖₁ (total-variation distance). τ = switch time. The post-shift
  window is τ … τ+Δ (Δ fixed at M7).
- **Drift scenarios** = abrupt, gradual, recurrent (`SIMULATION_DESIGN.md` §4).
  Concentration levels: focused (`κ = 0.1`) and spread (`κ = 1.0`).
- Static profile = cumulative mean. Adaptive profile = EWMA (sliding window as a
  variant). One hyperparameter setting per method, tuned on pooled tuning seeds
  (D-25).
- **Statistical protocol (D-21):** the unit is the seed-level mean, paired across
  methods (20 evaluation seeds). Two-sided Wilcoxon signed-rank test plus a paired
  bootstrap 95% CI, with Holm correction across H1–H6. A result is *practically
  meaningful* only if the whole CI lies beyond the SESOI: 3% relative for
  NDCG@K / expected utility, 2 interactions for adaptation lag, 0.02 absolute for
  mean TV error.

## Confirmatory hypotheses

| ID | RQ | Prediction | Disconfirmed if | Favours the proposed method? |
|---|---|---|---|---|
| **H1** Responsiveness | RQ1 | In drift scenarios, EWMA has (a) a shorter adaptation lag and (b) lower mean TV(z_t, ẑ_t) over the post-shift window than the cumulative mean | Not significant, or significantly favours the cumulative mean | Yes |
| **H2** Cost of adaptation | RQ1 | In the stable scenario, after burn-in, the cumulative mean has lower mean TV error than EWMA | EWMA is equal or better | No |
| **H3** Noise sensitivity | RQ1 | In the noisy scenario, EWMA's argmax ẑ_t changes more often while argmax z_t is unchanged (false switches) than the cumulative mean's does | Equal or fewer false switches | No |
| **H4** Adaptation → relevance | RQ1/RQ3 | In drift scenarios, Adaptive Top-K has higher post-shift NDCG@K (and expected utility) than Static Top-K. In the stable scenario the difference is below the SESOI | No significant post-shift gain; or a stable-scenario difference beyond the SESOI | Yes (drift) / neutral (stable) |
| **H5** Diversification trade-off | RQ2 | With the same profile, DPP gives higher category coverage than Top-K, and its NDCG@K cost relative to Top-K is smaller for spread than for focused interests | No coverage gain, or no (or reversed) interaction with concentration | Partly |
| **H6** Combined method | RQ3 | (a) In drift scenarios, Adaptive DPP is not Pareto-dominated in (NDCG@K, category coverage) by any compared method. (b) The adaptation × selector interaction on post-shift NDCG@K is tested two-sided: complementarity vs substitution | (a) Another method is better on both axes. (b) Reported in whichever direction it falls | Undetermined (b) |

## Exploratory (no directional prediction; not in the Holm family)
- EWMA vs sliding window at matched memory (`W ≈ 2/α − 1`).
- Sensitivity to `α`, `W`, `η`, `κ`, drift speed, `λ`, and the DPP trade-off.
- Adaptive calibration re-ranker (D-19) vs Adaptive DPP.
- Oracle-z methods (diagnostic ceiling) and oracle-tuned (per-scenario) settings.

## Mapping to experiments
| Hypothesis | Experiment (planned) | Result |
|---|---|---|
| H1, H2, H3 | M4 estimator study (tuning seeds); M8 confirmatory run (evaluation seeds) | — not run |
| H4, H5, H6 | M8 2×2 factorial runs | — not run |

## Amendments
*None.*
