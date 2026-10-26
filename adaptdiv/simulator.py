"""Open-loop interaction simulator (SIMULATION_DESIGN.md sections 5-8).

Generative model at step t (row k = t - 1):
    n_t ~ Bernoulli(eta)
    c_t ~ Categorical(z_t)            if n_t = 0
    c_t ~ Uniform{0..C-1}             if n_t = 1
    i_t ~ P(i | c_t, u)  proportional to  pi_i^gamma * exp(lambda * s_hat_i(u)),  i in c_t
Marginal:  P_t(i) = [(1 - eta) z_{t,c_i} + eta / C] * P(i | c_i, u).

Every draw uses inverse-CDF sampling on a per-user bank of uniforms that is shared by
all scenarios (common random numbers): sequences of two scenarios coincide wherever
their z_t and eta give the same outcome for the same uniform.

Leakage firewall (section 6): ``ObservedSequence`` holds only item ids; ``GroundTruth``
holds everything else and is for evaluation only.
"""
from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np

from . import seeding
from .catalogue import Catalogue, generate_catalogue
from .config import SCENARIOS, InteractionConfig, SimulationConfig
from .scenarios import (
    half_swap_permutation,
    noise_rate,
    sample_base_interest,
    shifted_interest,
    trajectory,
)


# --------------------------------------------------------------------------- records
@dataclass(frozen=True)
class ObservedSequence:
    """What an estimator/recommender may see: the item ids clicked, in order."""

    items: np.ndarray  # (T,) int, read-only

    def prefix(self, n: int) -> np.ndarray:
        """Items i_1..i_n (the history available when recommending for step n + 1)."""
        return self.items[:n]

    def __len__(self) -> int:
        return self.items.shape[0]


@dataclass(frozen=True)
class UserParams:
    """Hidden per-user parameters, shared across scenarios."""

    kappa: float
    z_a: np.ndarray       # (C,) base interest
    z_b: np.ndarray       # (C,) shifted interest
    permutation: np.ndarray  # (C,) half-swap permutation used for z_b
    taste: np.ndarray     # (d,) unit vector u


@dataclass(frozen=True)
class GroundTruth:
    """Everything the evaluation may use and the recommender must not."""

    scenario: str
    params: UserParams
    z: np.ndarray                 # (T, C) true interest trajectory
    noise_rate: float             # eta
    noise_flags: np.ndarray       # (T,) bool, n_t
    item_categories: np.ndarray   # (T,) int, c_t of each interaction
    item_conditional: np.ndarray  # (C, n_c) P(i | c, u), rows sum to 1

    def item_distribution(self, k: int) -> np.ndarray:
        """Exact P_t(i) over all N items for step t = k + 1 (0-based row k)."""
        C = self.z.shape[1]
        cat_mass = (1.0 - self.noise_rate) * self.z[k] + self.noise_rate / C
        return (cat_mass[:, None] * self.item_conditional).reshape(-1)


@dataclass(frozen=True)
class SimulatedUser:
    observed: ObservedSequence
    truth: GroundTruth


@dataclass(frozen=True)
class SimulatedData:
    """All data for one seed. ``users[(kappa, scenario)]`` lists users by index."""

    seed: int
    config: SimulationConfig
    catalogue: Catalogue
    train_item_counts: np.ndarray  # (N,) interaction counts of the training population
    users: dict[tuple[float, str], list[SimulatedUser]]


# --------------------------------------------------------------------------- pieces
def item_conditional(catalogue: Catalogue, taste: np.ndarray, icfg: InteractionConfig) -> np.ndarray:
    """(C, n_c) matrix of P(i | c, u) proportional to pi_i^gamma * exp(lambda * s_hat_i(u))."""
    s = catalogue.embeddings @ taste
    s_hat = (s - s.mean()) / s.std()
    log_w = icfg.gamma * np.log(catalogue.popularity) + icfg.taste_weight * s_hat
    C, n_c = catalogue.n_categories, catalogue.config.items_per_category
    log_w = log_w.reshape(C, n_c)
    w = np.exp(log_w - log_w.max(axis=1, keepdims=True))  # stable softmax per category
    return w / w.sum(axis=1, keepdims=True)


def _inverse_cdf(cdf_rows: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Index j with cdf[j-1] <= u < cdf[j] for each row; robust to cdf[-1] < 1 by rounding."""
    idx = (cdf_rows < u[:, None]).sum(axis=1)
    return np.minimum(idx, cdf_rows.shape[1] - 1)


def sample_interactions(
    z: np.ndarray,
    eta: float,
    cond: np.ndarray,
    bank: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample (items, categories, noise_flags) for a trajectory ``z`` (T, C).

    ``bank`` is a (3, T) array of uniforms: rows = noise, category, item draws.
    """
    T, C = z.shape
    n_c = cond.shape[1]
    u_noise, u_cat, u_item = bank[0, :T], bank[1, :T], bank[2, :T]

    noise = u_noise < eta
    signal_cat = _inverse_cdf(np.cumsum(z, axis=1), u_cat)
    noise_cat = np.minimum((u_cat * C).astype(int), C - 1)
    cats = np.where(noise, noise_cat, signal_cat)

    local = _inverse_cdf(np.cumsum(cond, axis=1)[cats], u_item)
    items = cats * n_c + local
    return items, cats, noise


def _user_params(rng: np.random.Generator, kappa: float, catalogue: Catalogue, magnitude: float) -> UserParams:
    """Draw z_A, the half-swap permutation and u (in that fixed order)."""
    z_a = sample_base_interest(kappa, catalogue.n_categories, rng)
    perm = half_swap_permutation(z_a, rng)
    z_b = shifted_interest(z_a, perm, magnitude)
    u = rng.standard_normal(catalogue.config.embedding_dim)
    return UserParams(kappa, z_a, z_b, perm, u / np.linalg.norm(u))


def _freeze(*arrays: np.ndarray) -> None:
    for a in arrays:
        a.setflags(write=False)


def simulate_user(
    params: UserParams,
    bank: np.ndarray,
    scenario: str,
    catalogue: Catalogue,
    cfg: SimulationConfig,
) -> SimulatedUser:
    """Generate one user's sequence for one scenario."""
    z = trajectory(scenario, params.z_a, params.z_b, cfg.scenario)
    eta = noise_rate(scenario, cfg.interaction.noise_rate, cfg.scenario)
    cond = item_conditional(catalogue, params.taste, cfg.interaction)
    items, cats, noise = sample_interactions(z, eta, cond, bank)
    _freeze(z, items, cats, noise, cond)
    truth = GroundTruth(scenario, params, z, eta, noise, cats, cond)
    return SimulatedUser(ObservedSequence(items), truth)


def train_item_counts(seed: int, catalogue: Catalogue, cfg: SimulationConfig) -> np.ndarray:
    """Item counts from the training population (D-26): stable interests, kappa levels alternating."""
    pop, kappas = cfg.population, cfg.population.kappas
    counts = np.zeros(catalogue.n_items, dtype=np.int64)
    for j in range(pop.train_users):
        params = _user_params(
            seeding.stream(seed, seeding.TRAINING, j, seeding.PARAMS),
            kappas[j % len(kappas)], catalogue, cfg.scenario.shift_magnitude,
        )
        bank = seeding.stream(seed, seeding.TRAINING, j, seeding.INTERACTIONS).random((3, pop.train_horizon))
        z = np.broadcast_to(params.z_a, (pop.train_horizon, catalogue.n_categories))
        cond = item_conditional(catalogue, params.taste, cfg.interaction)
        items, _, _ = sample_interactions(z, cfg.interaction.noise_rate, cond, bank)
        counts += np.bincount(items, minlength=catalogue.n_items)
    return counts


def smoothed_popularity(counts: np.ndarray) -> np.ndarray:
    """Add-one smoothed relative frequency r_i = (count_i + 1) / (sum + N); strictly positive."""
    return (counts + 1.0) / (counts.sum() + counts.shape[0])


def generate(seed: int, cfg: SimulationConfig, scenarios: tuple[str, ...] = SCENARIOS) -> SimulatedData:
    """Generate the catalogue, training counts and all evaluation users for one seed."""
    catalogue = generate_catalogue(cfg.catalogue, seeding.stream(seed, seeding.CATALOGUE))
    counts = train_item_counts(seed, catalogue, cfg)
    T = cfg.scenario.horizon

    users: dict[tuple[float, str], list[SimulatedUser]] = {}
    for k, kappa in enumerate(cfg.population.kappas):
        cell = {s: [] for s in scenarios}
        for j in range(cfg.population.users_per_cell):
            params = _user_params(
                seeding.stream(seed, seeding.USERS, k, j, seeding.PARAMS),
                kappa, catalogue, cfg.scenario.shift_magnitude,
            )
            _freeze(params.z_a, params.z_b, params.permutation, params.taste)
            bank = seeding.stream(seed, seeding.USERS, k, j, seeding.INTERACTIONS).random((3, T))
            for s in scenarios:
                cell[s].append(simulate_user(params, bank, s, catalogue, cfg))
        for s in scenarios:
            users[(kappa, s)] = cell[s]
    return SimulatedData(seed, cfg, catalogue, counts, users)


# Field names the firewall test checks against.
OBSERVED_FIELDS = frozenset(f.name for f in fields(ObservedSequence))
