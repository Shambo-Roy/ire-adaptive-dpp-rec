# Literature matrix — Milestone 1 (compiled 2026-10-26)

## How to read this file

**Verification levels** (column "Ver."):
- **M**: bibliographic metadata (authors, title, venue, year, pages/DOI) checked against
  at least one primary index (publisher, ACM DL, DBLP or Webis anthology, NeurIPS/ICML
  proceedings, arXiv).
- **A**: the claims in the "Findings" column come from the abstract or the
  publisher's summary.
- **F**: full text read. **No entry is at level F yet.** Papers whose algorithmic
  details we will implement (Chen et al. 2018, Kulesza & Taskar 2012, Steck 2018)
  must be read in full before Milestone 6, and their rows upgraded.
- **S**: some claims come from secondary sources (surveys, slides). Such claims must
  not be cited as the paper's own until they reach level F.

Nothing here should be cited in the report beyond what its verification level
supports. "Relevance to us" is our own interpretation, not a claim made by the paper.

---

## A. Determinantal point processes (DPPs) and DPP-based recommendation

| # | Reference | Ver. | Method / contribution | Data | Metrics | Findings (as stated) | Relevance to us |
|---|---|---|---|---|---|---|---|
| A1 | Kulesza, A. & Taskar, B. (2012). *Determinantal Point Processes for Machine Learning.* Foundations and Trends in ML 5(2–3):123–286. doi:10.1561/2200000044. arXiv:1207.6083 | M, A | Monograph: DPP definitions, the L-ensemble, quality/diversity decomposition, exact sampling, marginalisation, conditioning, learning | Summarisation, search, pose, timelines | — | DPPs give exact, efficient algorithms for sampling, marginalisation and conditioning; a similarity kernel induces repulsion | Primary source for `L = Diag(q) S Diag(q)`, PSD requirements, and the quality vs diversity interpretation |
| A2 | Kulesza, A. & Taskar, B. (2011). *k-DPPs: Fixed-Size Determinantal Point Processes.* ICML 2011, pp. 1193–1200 | M, A | DPP conditioned on set size k; normalisation, sampling and marginalisation algorithms | Image search | Human judgements | k-DPPs outperformed MMR in human judgements on image search | We need fixed-size lists (size K). This is an alternative to greedy MAP (sample instead of maximise) |
| A3 | Ko, C.-W., Lee, J. & Queyranne, M. (1995). *An Exact Algorithm for Maximum Entropy Sampling.* Operations Research 43(4):684–691. doi:10.1287/opre.43.4.684 | M, A | Shows that maximising the determinant of a covariance submatrix is NP-hard; branch-and-bound exact algorithm | Environmental monitoring covariances | — | NP-hardness of max-det subset selection | Justifies calling greedy DPP MAP an approximation |
| A4 | Gillenwater, J., Kulesza, A. & Taskar, B. (2012). *Near-Optimal MAP Inference for Determinantal Point Processes.* NIPS 25, pp. 2735–2743 | M, A | Continuous relaxation; 1/4-approximation for a class of non-monotone DPPs | Synthetic and real | Log-probability | Reported to outperform greedy | Shows that greedy is not optimal. We use greedy for speed and state this limitation |
| A5 | Chen, L., Zhang, G. & Zhou, E. (2018). *Fast Greedy MAP Inference for Determinantal Point Process to Improve Recommendation Diversity.* NeurIPS 2018, pp. 5622–5633. arXiv:1709.05135 (third author listed as Hanning Zhou on arXiv) | M, A | Incremental-Cholesky greedy MAP, much faster than naive greedy; sliding-window variant for long sequences | Several public recsys datasets, online A/B test | Relevance and diversity metrics; runtime | Reported faster than competitors and a better relevance–diversity trade-off | **Primary algorithm candidate (D-15).** The relevance–diversity trade-off parametrisation of the kernel must be checked in the full text before M6 |
| A6 | Wilhelm, M., Ramanathan, A., Bonomo, A., Jain, S., Chi, E. H. & Gillenwater, J. (2018). *Practical Diversified Recommendations on YouTube with Determinantal Point Processes.* CIKM 2018, pp. 2165–2173. doi:10.1145/3269206.3272018 | M | Industrial DPP deployment for feed diversification | YouTube (proprietary) | Online metrics | Abstract not retrieved; findings not recorded yet | Evidence that DPP re-ranking is used in practice. Cite only after reading |
| A7 | Liu, Y., Walder, C. & Xie, L. (2022). *Determinantal Point Process Likelihoods for Sequential Recommendation.* SIGIR 2022. arXiv:2204.11562 | M, A | DPP-likelihood training losses with quality/diversity kernel decomposition | Three real datasets | Quality and diversity | Reported improvements on both quality and diversity | DPP used in *sequential* recommendation, at training time rather than at re-ranking time |
| A8 | Liu, Y., Zhang, Y., Wu, Q., Miao, C., Cui, L., Zhao, B., Zhao, Y. & Guan, L. (2019). *Diversity-Promoting Deep Reinforcement Learning for Interactive Recommendation.* arXiv:1903.07826 (preprint; venue not verified) | M (arXiv only), A | D²RL: an actor network outputs parameters of a user-specific DPP kernel that adapts over interactions | Not recorded | Not recorded | DPP kernel adapted to changing preferences via RL | **Closest prior work on time-varying DPP quality.** It differs from ours by learning adaptation with RL in an interactive (closed-loop) setting |
| A9 | Ibrahim, C., Bederina, H., Cuesta, D., Montier, L., Delabre, C. & Vie, J.-J. (2025). *Diversified recommendations of cultural activities with personalized determinantal point processes.* arXiv:2509.10392 (RecSoGood workshop @ RecSys 2025) | M, A | Personalised DPP: quality–diversity decomposition with user-preference-weighted quality and a trade-off parameter | pass Culture (France) | Offline and online | Relevance–diversity trade-offs studied | Shows that **personalised quality inside a DPP kernel is not new**. Our contribution can't be "interest-weighted DPP" alone |

## B. Diversity, calibration and intent-aware ranking

| # | Reference | Ver. | Method / contribution | Data | Metrics | Findings (as stated) | Relevance to us |
|---|---|---|---|---|---|---|---|
| B1 | Carbonell, J. & Goldstein, J. (1998). *The Use of MMR, Diversity-Based Reranking for Reordering Documents and Producing Summaries.* SIGIR '98, pp. 335–336. doi:10.1145/290941.291025 | M, A | Maximal Marginal Relevance: greedy relevance minus redundancy | Retrieval, summarisation | — | Clearest benefit in multi-document summarisation | A classic, cheaper diversification baseline (optional in D-16) |
| B2 | Ziegler, C.-N., McNee, S. M., Konstan, J. A. & Lausen, G. (2005). *Improving Recommendation Lists Through Topic Diversification.* WWW 2005, pp. 22–32. doi:10.1145/1060745.1060754 | M, A | Topic diversification; intra-list similarity (ILS) metric | Book ratings (BookCrossing); user study with >2,100 participants | ILS, accuracy, satisfaction | Diversification lowered accuracy but raised user satisfaction | Origin of the intra-list similarity/diversity metric. Shows the accuracy–diversity trade-off |
| B3 | Agrawal, R., Gollapudi, S., Halverson, A. & Ieong, S. (2009). *Diversifying Search Results.* WSDM 2009, pp. 5–14. doi:10.1145/1498759.1498766 | M, A | Taxonomy-based diversification for ambiguous queries; intent-aware versions of NDCG, MRR and MAP | Web search | Intent-aware metrics | Their method scored higher on the intent-aware metrics | Categories as "intents" with a distribution, which parallels our `z_t`. Intent-aware metrics are a possible evaluation choice |
| B4 | Clarke, C. L. A., Kolla, M., Cormack, G. V., Vechtomova, O., Ashkan, A., Büttcher, S. & MacKinnon, I. (2008). *Novelty and Diversity in Information Retrieval Evaluation.* SIGIR 2008, pp. 659–666. doi:10.1145/1390334.1390446 | M, A/S | α-nDCG (the attribution comes from secondary sources) | TREC QA-derived collection | α-nDCG | Distinguishes novelty from diversity | Candidate diversity-aware relevance metric (D-17) |
| B5 | Vargas, S., Castells, P. & Vallet, D. (2011). *Intent-Oriented Diversity in Recommender Systems.* SIGIR 2011 (poster), pp. 1211–1212. doi:10.1145/2009916.2010124 | M, A | Brings IR intent-aware diversification to recommendation via "user profile aspects" | Not recorded | Intent-aware metrics | Reported empirical support | Directly analogous: user-interest aspects that weight diversification |
| B6 | Teo, C. H., Nassif, H., Hill, D., Srinivasan, S., Goodman, M., Mohan, V. & Vishwanathan, S. V. N. (2016). *Adaptive, Personalized Diversity for Visual Discovery.* RecSys 2016. doi:10.1145/2959100.2959171. arXiv:1810.01477 | M, A | Bayesian relevance plus submodular category diversification with **personalised category preferences learned from behaviour** | Amazon Stream (live traffic) | CTR, session duration | Reported lift in CTR and session duration | Closest industrial analogue of "interest-weighted diversification", using submodular re-ranking rather than a DPP |
| B7 | Steck, H. (2018). *Calibrated Recommendations.* RecSys 2018, pp. 154–162. doi:10.1145/3240323.3240372 | M, S | Re-ranking so that the list's genre distribution matches the user's history distribution (KL divergence, greedy submodular surrogate) | Movie ratings (secondary sources) | KL calibration | Accuracy-only ranking can crowd out minor interests (from slides and survey, level S) | **Strong competitor idea**: matches list composition to an interest distribution. Raises decision D-19 (calibration baseline). Must be read in full |
| B8 | Lin, K., Mansoury, M., Eskandanian, F., Sabouri, M. & Mobasher, B. (2024). *Beyond Static Calibration: The Impact of User Preference Dynamics on Calibrated Recommendation.* UMAP Adjunct '24, pp. 86–91. doi:10.1145/3631700.3664869. arXiv:2405.10232 | M, A | Calibrates against recent time windows of varying length instead of the full history | Several real domains | Calibration | Effect differs by domain; described as preliminary and exploratory | **Closest conceptual work**: static vs time-windowed interest profiles for list composition. That is our static vs adaptive factor, applied to calibration instead of a DPP |
| B9 | Kaminskas, M. & Bridge, D. (2016/2017). *Diversity, Serendipity, Novelty, and Coverage: A Survey and Empirical Analysis of Beyond-Accuracy Objectives in Recommender Systems.* ACM TiiS 7(1). doi:10.1145/2926720 (year differs between sources) | M, A | Survey of definitions and metrics plus offline experiments | Several | ILD, coverage, etc. | Optimising one objective affects the others | Reference for metric definitions (ILD as average pairwise distance) |
| B10 | Wu, H., Zhang, Y., Ma, C., Lyu, F., He, B., Mitra, B. & Liu, X. (2024). *Result Diversification in Search and Recommendation: A Survey.* IEEE TKDE. doi:10.1109/TKDE.2024.3382262. arXiv:2212.14464 | M, A | Unified taxonomy of diversification metrics and methods | — | — | Notes that the field is fragmented | Survey for the related-work section |

## C. Session-based / sequential recommendation and multi-interest models

| # | Reference | Ver. | Method / contribution | Data | Metrics | Findings (as stated) | Relevance to us |
|---|---|---|---|---|---|---|---|
| C1 | Hidasi, B., Karatzoglou, A., Baltrunas, L. & Tikk, D. (2016). *Session-based Recommendations with Recurrent Neural Networks.* ICLR 2016. arXiv:1511.06939 | M, A | GRU4Rec: RNN over sessions with a ranking loss | Two datasets | Recall@20, MRR@20 | Gains over common baselines | Context for session-based recommendation. We deliberately don't use neural models (no justification for one in a controlled simulation) |
| C2 | Ludewig, M. & Jannach, D. (2018). *Evaluation of Session-based Recommendation Algorithms.* UMUAI 28(4–5):331–390. doi:10.1007/s11257-018-9209-6 | M, A | Large benchmark of session-based algorithms | Multiple | Multiple | The 2021 follow-up (Ludewig et al., UMUAI 31) reports that simple nearest-neighbour heuristics are often preferable | Supports simple, transparent relevance models. A reason not to need a neural base ranker |
| C3 | Wang, S., Cao, L., Wang, Y., Sheng, Q. Z., Orgun, M. A. & Lian, D. (2021). *A Survey on Session-based Recommender Systems.* ACM Computing Surveys 54(7), Art. 154. doi:10.1145/3465401 | M, A | Problem statement and taxonomy for session-based recommender systems | — | — | These systems target short-term, dynamic preferences | Definitions for the introduction |
| C4 | Cen, Y., Zhang, J., Zou, X., Zhou, C., Yang, H. & Tang, J. (2020). *Controllable Multi-Interest Framework for Recommendation.* KDD 2020. arXiv:2005.09347 | M, A | ComiRec: several interest embeddings per user; aggregation step with a controllable accuracy–diversity factor (greedy) | Amazon, Taobao | Recall, NDCG, HR, diversity | A single user embedding can't represent multiple interests | Multi-interest representation is the neural counterpart of our `z_t` mixture. Also has an accuracy–diversity knob |
| C5 | Rohde, D. & Bonner, S. (2019). *Latent Variable Session-Based Recommendation.* arXiv:1904.10784 (preprint) | M (arXiv), A | Probabilistic latent-variable model for sessions | Not confirmed | Not confirmed | Competitive against baselines | Latent-interest framing of sessions |

## D. Temporal dynamics, interest drift and concept drift

| # | Reference | Ver. | Method / contribution | Data | Metrics | Findings (as stated) | Relevance to us |
|---|---|---|---|---|---|---|---|
| D1 | Ding, Y. & Li, X. (2005). *Time Weight Collaborative Filtering.* CIKM 2005, pp. 485–492. doi:10.1145/1099554.1099689 | M, A | Time-decay weighting of ratings in item-based CF, with a personalised decay rate per item cluster | Not recorded | Precision | Improved precision without extra complexity order | **Precedent for exponential forgetting of user history** (the idea behind EWMA) |
| D2 | Koren, Y. (2009). *Collaborative Filtering with Temporal Dynamics.* KDD 2009 | M, A | timeSVD++; temporal neighbourhood model with exponential decay | Netflix | RMSE | Improved on prior Netflix results | Classic evidence that preferences drift and that decay helps |
| D3 | Gama, J., Žliobaitė, I., Bifet, A., Pechenizkiy, M. & Bouchachia, A. (2014). *A Survey on Concept Drift Adaptation.* ACM Computing Surveys 46(4), Art. 44. doi:10.1145/2523813 | M, A | Taxonomy of drift (sudden, gradual, incremental, recurring) and adaptation strategies (forgetting, windows, detectors) | — | Evaluation of adaptive learners | — | Vocabulary for our drift scenarios. Justifies EWMA (fading factor) and windows as standard forgetting mechanisms |
| D4 | Bifet, A. & Gavaldà, R. (2007). *Learning from Time-Changing Data with Adaptive Windowing.* SDM 2007, pp. 443–448. doi:10.1137/1.9781611972771.42 | M, A | ADWIN: window size adapts to the detected rate of change, with false-positive/negative bounds | Synthetic and real streams | — | Removes the need to guess a time scale | Candidate optional adaptive-rate estimator (an extension, not in the core plan) |

## E. Simulation for recommender-systems research

| # | Reference | Ver. | Method / contribution | Relevance to us |
|---|---|---|---|---|
| E1 | Ie, E., Hsu, C., Mladenov, M., Jain, V., Narvekar, S., Wang, J., Wu, R. & Boutilier, C. (2019). *RecSim: A Configurable Simulation Platform for Recommender Systems.* arXiv:1909.04847 | M, A | Configurable latent user state, dynamics and choice models; aimed at reinforcement learning (closed loop) | Shows that latent-state user simulators are an accepted research tool. Ours is simpler and open-loop |
| E2 | Stavinova, E., Grigorievskiy, A., Volodkevich, A., Chunaev, P., Bochenina, K. & Bugaychenko, D. (2022). *Synthetic Data-Based Simulators for Recommender Systems: A Survey.* arXiv:2206.11338 (venue not verified) | M (arXiv), A | Classifies simulators; discusses validation and the gap to real-world behaviour | Supports our threats-to-validity discussion: simulator assumptions limit external validity |

---

## Synthesis

### What is established (supported by the sources above at level M/A)
1. DPPs give a principled quality × diversity set objective, and greedy MAP is
   the standard fast approximation. Exact MAP is NP-hard (A1, A3, A5).
2. DPP re-ranking for recommendation diversity is established in research and
   industry (A5, A6), including personalised quality terms (A9) and kernels adapted
   over interactions with RL (A8).
3. Weighting diversification by the user's interests or intents is established
   (B3, B5, B6, B7).
4. User preferences drift, and forgetting mechanisms (exponential decay, windows)
   are standard responses (D1–D4). Recent work questions *static* profiles for
   list composition (B8).
5. Diversification usually costs some accuracy. Whether it helps users depends on
   context (B2, B9).

### The gap we investigate (stated cautiously)
In the sources reviewed, **we did not find** a *controlled* study that:
(a) uses ground-truth latent interests to measure estimation error and adaptation
lag directly, and
(b) separates, with a factorial design, the effect of *adaptive* vs *static*
interest estimation from the effect of *DPP* vs *Top-K* selection, and their
interaction, across *different drift types* (abrupt, gradual, recurrent, noisy) and
interest concentrations.

The closest works either adapt the profile but use calibration rather than a DPP,
on real data without ground truth (B8), or adapt the DPP kernel in an RL closed loop
without isolating the components (A8).

### Claims we cannot make
- **No novelty claim for the method.** Interest-weighted DPP (A9, A8),
  personalised diversification (B6) and exponential forgetting (D1, D2) all exist.
  At most the contribution is a **controlled empirical analysis** of how these
  existing components interact under intent drift.
- "We did not find" is not "it does not exist". The search covered 30 sources
  (plus one follow-up noted under C2) in one milestone. It was a targeted search,
  not a systematic review.
- Synthetic results must not be presented as evidence of real-world effectiveness
  (E2).

### Consequences for design (proposals, not yet approved)
- Add a **calibration (Steck-style) re-ranker** as a non-DPP, interest-aware
  diversification baseline (D-19). Otherwise a reviewer can ask whether the DPP is
  needed at all.
- Report results by **interest concentration** (focused vs spread). The B2/B9
  trade-off suggests that DPP's relevance cost depends on it.
- Use drift-type vocabulary from D3 (sudden, gradual, recurring) for the scenarios.

## Reading list before Milestone 6 (to reach level F)
A1 (§2–4: L-ensembles, quality/diversity decomposition), A5 (algorithm and trade-off
kernel), B7 (calibration objective and greedy algorithm), A6, A9, B8.
