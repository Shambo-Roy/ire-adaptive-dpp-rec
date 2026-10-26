"""Catalogue construction, configuration round-trip and seed protocol."""
import numpy as np

from adaptdiv import seeding
from adaptdiv.catalogue import generate_catalogue
from adaptdiv.config import CatalogueConfig, SimulationConfig


def _cat(seed=0, **kw):
    return generate_catalogue(CatalogueConfig(**kw), seeding.stream(seed, seeding.CATALOGUE))


def test_catalogue_shapes_and_grouping():
    c = _cat()
    assert c.embeddings.shape == (500, 32)
    assert np.array_equal(c.categories, np.repeat(np.arange(10), 50))
    assert np.allclose(np.linalg.norm(c.embeddings, axis=1), 1.0)
    assert np.allclose(np.linalg.norm(c.centroids, axis=1), 1.0)


def test_popularity_positive_and_normalised_per_category():
    c = _cat()
    assert (c.popularity > 0).all()
    assert np.allclose(c.popularity.reshape(10, 50).sum(axis=1), 1.0)


def test_embedding_similarity_matches_design_evidence():
    """SIMULATION_DESIGN.md section 2: sigma=1 gives cosine about 0.50 within, 0.00 across categories."""
    within, across = [], []
    for seed in range(10):
        c = _cat(seed)
        S = c.embeddings @ c.embeddings.T
        same = c.categories[:, None] == c.categories[None, :]
        off = ~np.eye(c.n_items, dtype=bool)
        within.append(S[same & off].mean())
        across.append(S[~same].mean())
    assert abs(np.mean(within) - 0.50) < 0.03
    assert abs(np.mean(across)) < 0.03


def test_public_view_hides_popularity():
    pub = _cat().public()
    assert not hasattr(pub, "popularity")
    assert not hasattr(pub, "centroids")


def test_catalogue_arrays_read_only():
    c = _cat()
    for a in (c.embeddings, c.popularity, c.categories):
        assert not a.flags.writeable


def test_same_seed_same_catalogue_different_seed_differs():
    a, b, other = _cat(3), _cat(3), _cat(4)
    assert np.array_equal(a.embeddings, b.embeddings) and np.array_equal(a.popularity, b.popularity)
    assert not np.allclose(a.embeddings, other.embeddings)


def test_streams_depend_only_on_seed_and_path():
    x = seeding.stream(7, seeding.USERS, 1, 5, seeding.PARAMS).random(4)
    y = seeding.stream(7, seeding.USERS, 1, 5, seeding.PARAMS).random(4)
    z = seeding.stream(7, seeding.USERS, 1, 6, seeding.PARAMS).random(4)
    assert np.array_equal(x, y)
    assert not np.array_equal(x, z)


def test_config_json_round_trip(tmp_path):
    cfg = SimulationConfig()
    path = tmp_path / "cfg.json"
    cfg.save_json(path)
    assert SimulationConfig.load_json(path) == cfg
