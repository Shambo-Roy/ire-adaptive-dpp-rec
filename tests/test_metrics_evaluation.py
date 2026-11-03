"""Metrics (hand-checked), evaluator (independent recomputation), leakage and fairness checks."""
import ast
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

import adaptdiv
from adaptdiv import metrics as M
from adaptdiv.config import PopulationConfig, SimulationConfig
from adaptdiv.estimators import PROPOSED_ESTIMATOR, categories_of
from adaptdiv.evaluation import (
    EvalConfig,
    build_profiles,
    evaluate,
    list_metrics,
    method_lists,
    post_mask,
    scored_steps,
    switch_events,
)
from adaptdiv.recommenders import Method, popularity_list, standard_methods
from adaptdiv.relevance import global_popularity, within_category_popularity
from adaptdiv.similarity import similarity_matrix
from adaptdiv.simulator import generate


# ------------------------------------------------------------------ hand-checked metrics
def test_target_matrix():
    assert M.target_matrix(np.arange(10), np.array([2, 5]), 3).tolist() == [[2, 3, 4], [5, 6, 7]]


def test_ndcg_hand_example():
    """Hits at ranks 1 and 3 of 3 with 2 targets: (1 + 1/2) / (1 + 1/log2(3)) = 0.91972."""
    assert np.isclose(M.ndcg_at_k(np.array([[True, False, True]]), np.array([2]))[0], 1.5 / (1 + 1 / np.log2(3)))


def test_ndcg_perfect_zero_and_ideal_capped_at_k():
    assert np.isclose(M.ndcg_at_k(np.array([[True, True]]), np.array([5]))[0], 1.0)  # 5 targets, K = 2
    assert M.ndcg_at_k(np.array([[False, False]]), np.array([2]))[0] == 0.0
    assert np.isnan(M.ndcg_at_k(np.array([[False, False]]), np.array([0]))[0])


def test_rank_matters_for_ndcg_not_recall():
    top = np.array([[True, False, False]])
    low = np.array([[False, False, True]])
    n = np.array([1])
    assert M.ndcg_at_k(top, n)[0] > M.ndcg_at_k(low, n)[0]
    assert M.recall_at_k(top, n)[0] == M.recall_at_k(low, n)[0] == 1.0


def test_repeated_targets_count_once():
    targets = np.array([[3, 3, 4, 3]])
    assert M.n_targets(targets)[0] == 2
    hit = M.hits(np.array([[3, 9]]), targets)
    assert hit.tolist() == [[True, False]]
    assert np.isclose(M.recall_at_k(hit, M.n_targets(targets))[0], 0.5)


def test_valid_mask_restricts_targets():
    targets = np.array([[3, 3, 4]])
    valid = np.array([[False, False, True]])  # item 3 already seen
    assert M.n_targets(targets, valid)[0] == 1
    assert M.hits(np.array([[3, 4]]), targets, valid).tolist() == [[False, True]]


def test_expected_utility_with_mask():
    probs = np.array([[0.2, 0.3]])
    assert np.isclose(M.expected_utility(probs)[0], 0.5)
    assert np.isclose(M.expected_utility(probs, np.array([[True, False]]))[0], 0.2)


def test_category_coverage():
    assert M.category_coverage(np.array([[1, 1, 2, 5]]), 10)[0] == 3 / 4   # min(K, C) = 4
    assert M.category_coverage(np.array([[0, 0, 1, 1]]), 2)[0] == 1.0


def test_intra_list_diversity():
    assert np.isclose(M.intra_list_diversity(np.eye(3)[None])[0], 1.0)
    same = np.tile(np.array([1.0, 0.0]), (3, 1))[None]
    assert np.isclose(M.intra_list_diversity(same)[0], 0.0)
    one_pair = np.array([[[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]])  # cosines 1, 0, 0
    assert np.isclose(M.intra_list_diversity(one_pair)[0], 2 / 3)


# ------------------------------------------------------------------ evaluator
@pytest.fixture(scope="module")
def tiny():
    cfg = replace(SimulationConfig(), population=PopulationConfig(users_per_cell=3, train_users=50))
    return generate(0, cfg)


EV = EvalConfig(stride=25)


def _context(data):
    pub = data.catalogue.public()
    r = within_category_popularity(data.train_item_counts, pub.categories, pub.n_categories)
    return pub, r, similarity_matrix(pub.embeddings), popularity_list(global_popularity(data.train_item_counts), 10)


def test_scored_steps_have_full_target_windows():
    steps = scored_steps(200, EvalConfig())
    assert steps[0] == 20 and steps[-1] == 195
    assert (steps + 5 <= 200).all()


def test_post_mask_alignment():
    steps = np.arange(95, 155)
    mask = post_mask(steps, [100], 50)
    assert steps[mask][0] == 99 and steps[mask][-1] == 148
    assert mask.sum() == 50


def test_switch_events():
    cfg = SimulationConfig()
    assert [e[0] for e in switch_events("abrupt", cfg)] == [100]
    assert [e[0] for e in switch_events("gradual", cfg)] == [100]
    assert [e[0] for e in switch_events("recurrent", cfg)] == [70, 140]
    assert switch_events("stable", cfg) == []


def test_list_metrics_match_independent_slow_computation(tiny):
    pub, r, S, pop = _context(tiny)
    user = tiny.users[(0.1, "abrupt")][1]
    cats = categories_of(user.observed.items, pub.categories)[None]
    prof = build_profiles(cats, pub.n_categories, None)["adaptive"][0]
    steps = np.array([30, 99, 150])
    lists = method_lists(Method("m", "adaptive", 0.6), prof, r, pop, pub, S, steps, 10)
    got = list_metrics(lists, steps, user, pub, EvalConfig())

    items = user.observed.items
    disc = 1 / np.log2(np.arange(2, 12))
    for n, t in enumerate(steps):
        lst = lists[n].tolist()
        seen = set(items[:t].tolist())
        tg = set(items[t:t + 5].tolist())
        new = tg - seen
        dcg = sum(disc[p] for p, i in enumerate(lst) if i in tg)
        assert np.isclose(got["ndcg"][n], dcg / disc[:len(tg)].sum())
        assert np.isclose(got["recall"][n], len(tg & set(lst)) / len(tg))
        P = user.truth.item_distribution(t)
        assert np.isclose(got["eu"][n], P[lst].sum())
        assert np.isclose(got["eu_new"][n], sum(P[i] for i in lst if i not in seen))
        assert np.isclose(got["seen"][n], np.mean([i in seen for i in lst]))
        assert np.isclose(got["coverage"][n], len({pub.categories[i] for i in lst}) / 10)
        z = user.truth.z[t]                                        # z_{t+1}, the interest the list serves
        q = [sum(pub.categories[i] == g for i in lst) / 10 for g in range(10)]
        kl = sum(z[g] * np.log(z[g] / (0.99 * q[g] + 0.01 * z[g])) for g in range(10) if z[g] > 0)
        assert np.isclose(got["miscal"][n], kl)
        if new:
            dcg_new = sum(disc[p] for p, i in enumerate(lst) if i in new)
            assert np.isclose(got["ndcg_new"][n], dcg_new / disc[:len(new)].sum())
        else:
            assert np.isnan(got["ndcg_new"][n])


def test_dpp_lists_are_sorted_by_score(tiny):
    pub, r, S, pop = _context(tiny)
    user = tiny.users[(1.0, "stable")][0]
    prof = build_profiles(categories_of(user.observed.items, pub.categories)[None], pub.n_categories, None)["adaptive"][0]
    t = np.array([50])
    lst = method_lists(Method("m", "adaptive", 0.5), prof, r, pop, pub, S, t, 10)[0]
    scores = r[lst] * (prof[50][pub.categories[lst]] + 0.01)
    assert (np.diff(scores) <= 1e-12).all()


def test_evaluate_records_are_complete_and_paired(tiny):
    methods = standard_methods(0.6)
    rec = evaluate(tiny, methods, EV)
    for scen in ("stable", "abrupt", "gradual", "recurrent", "noisy"):
        for kappa in (0.1, 1.0):
            for m in methods:
                assert np.isfinite(rec[(kappa, scen, m.name, "overall", "ndcg")])
                assert ((kappa, scen, m.name, "post", "ndcg") in rec) == (scen in ("abrupt", "gradual", "recurrent"))
            assert (kappa, scen, "est:ewma", "overall", "tracking") in rec
    assert (0.1, "abrupt", "est:cumulative", "post", "lag") in rec


def test_theta_one_dpp_equals_top_k_records(tiny):
    a = Method("as_dpp", "adaptive", 1.0)
    b = Method("adaptive_topk", "adaptive")
    rec = evaluate(tiny, (a, b), EV)
    for key, value in rec.items():
        if key[2] == "as_dpp":
            assert value == rec[(key[0], key[1], "adaptive_topk", key[3], key[4])]


def test_extra_profile_matches_the_same_built_in_estimator(tiny):
    """An extra profile with the proposed spec gives the proposed records, and changes nothing else."""
    base = Method("adaptive_topk", "adaptive")
    extra = Method("copy", "copy")
    plain = evaluate(tiny, (base,), EV)
    rec = evaluate(tiny, (base, extra), EV, {"copy": PROPOSED_ESTIMATOR})
    for key, value in plain.items():
        assert rec[key] == value
        if key[2] == "adaptive_topk":
            assert rec[(key[0], key[1], "copy", key[3], key[4])] == value



def test_calibrated_method_adds_records_and_changes_nothing_else(tiny):
    """D-19: adding a calibrated method leaves every other record identical; lambda = 0 equals Top-K."""
    base = Method("adaptive_topk", "adaptive")
    plain = evaluate(tiny, (base,), EV)
    rec = evaluate(tiny, (base, Method("cal0", "adaptive", calib_lambda=0.0), Method("cal", "adaptive", calib_lambda=0.99)), EV)
    for key, value in plain.items():
        assert rec[key] == value
        if key[2] == "adaptive_topk":
            assert rec[(key[0], key[1], "cal0", key[3], key[4])] == value
    assert rec[(0.1, "stable", "cal", "overall", "miscal")] < rec[(0.1, "stable", "adaptive_topk", "overall", "miscal")]


# ------------------------------------------------------------------ leakage
def test_recommendation_code_cannot_import_ground_truth():
    """Estimators, relevance, similarity, DPP and recommenders must not import the simulator."""
    root = Path(adaptdiv.__file__).parent
    for name in ("estimators", "relevance", "similarity", "dpp", "recommenders", "calibration"):
        tree = ast.parse((root / f"{name}.py").read_text(encoding="utf-8"))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                imported.add(node.module or "")
            elif isinstance(node, ast.Import):
                imported.update(a.name for a in node.names)
        assert not {"simulator", "scenarios", "catalogue"} & {m.split(".")[-1] for m in imported}, name


def test_non_oracle_profiles_contain_no_truth_without_opt_in(tiny):
    pub = tiny.catalogue.public()
    cats = categories_of(tiny.users[(0.1, "abrupt")][0].observed.items, pub.categories)[None]
    assert "oracle" not in build_profiles(cats, pub.n_categories, None)


def test_list_at_t_is_unchanged_by_future_clicks(tiny):
    """Replace every click after t: profiles and lists up to t stay identical, and targets differ."""
    pub, r, S, pop = _context(tiny)
    items = tiny.users[(1.0, "abrupt")][0].observed.items
    t = 120
    changed = items.copy()
    changed[t:] = (changed[t:] + 7) % pub.categories.shape[0]
    steps = np.array([30, 80, 120])
    lists = []
    for seq in (items, changed):
        prof = build_profiles(categories_of(seq, pub.categories)[None], pub.n_categories, None)
        lists.append(method_lists(Method("m", "adaptive", 0.6), prof["adaptive"][0], r, pop, pub, S, steps, 10))
    assert np.array_equal(lists[0], lists[1])
    assert not np.array_equal(M.target_matrix(items, np.array([t]), 5), M.target_matrix(changed, np.array([t]), 5))


def test_oracle_row_is_the_step_it_predicts(tiny):
    user = tiny.users[(0.1, "abrupt")][0]
    prof = build_profiles(categories_of(user.observed.items, tiny.catalogue.public().categories)[None],
                          10, user.truth.z[None])["oracle"][0]
    assert np.array_equal(prof[99], user.truth.params.z_b)  # row t = 99 predicts step 100, the first z_B step
    assert np.array_equal(prof[98], user.truth.params.z_a)
