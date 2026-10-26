"""Simulation configuration (SIMULATION_DESIGN.md).

All simulator parameters live here as plain dataclasses so that a run's exact settings
can be written to / read from JSON and stored next to its results. Defaults are the
values approved in Milestone 2 (DECISIONS.md D-08..D-10, D-22..D-26).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class CatalogueConfig:
    n_categories: int = 10          # C
    items_per_category: int = 50    # n_c  (N = C * n_c)
    embedding_dim: int = 32         # d   (must be >= K, D-12)
    sigma: float = 1.0              # within-category embedding spread
    popularity_spread: float = 1.0  # s_pop, log-normal popularity spread

    @property
    def n_items(self) -> int:
        return self.n_categories * self.items_per_category


@dataclass(frozen=True)
class InteractionConfig:
    gamma: float = 1.0      # popularity exponent
    taste_weight: float = 0.5  # lambda, weight of the hidden user taste (D-09)
    noise_rate: float = 0.05   # eta in all scenarios except "noisy"


@dataclass(frozen=True)
class ScenarioConfig:
    horizon: int = 200              # T interactions per user
    abrupt_switch: int = 100        # tau (first step with z_B)
    gradual_start: int = 80         # tau_0 (ramp starts after this step)
    gradual_width: int = 40         # W (ramp length)
    recurrent_switches: tuple[int, int] = (70, 140)  # tau_1, tau_2
    noisy_noise_rate: float = 0.3   # eta in the "noisy" scenario
    shift_magnitude: float = 1.0    # m: z_B(m) = (1-m) z_A + m * permuted z_A


SCENARIOS: tuple[str, ...] = ("stable", "abrupt", "gradual", "recurrent", "noisy")


@dataclass(frozen=True)
class PopulationConfig:
    kappas: tuple[float, ...] = (0.1, 1.0)  # Dirichlet concentration levels (D-23)
    users_per_cell: int = 100               # U per (seed, kappa, scenario)
    train_users: int = 1000                 # U_train (D-26)
    train_horizon: int = 50                 # interactions per training user


@dataclass(frozen=True)
class SimulationConfig:
    catalogue: CatalogueConfig = field(default_factory=CatalogueConfig)
    interaction: InteractionConfig = field(default_factory=InteractionConfig)
    scenario: ScenarioConfig = field(default_factory=ScenarioConfig)
    population: PopulationConfig = field(default_factory=PopulationConfig)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "SimulationConfig":
        sc = dict(d["scenario"])
        sc["recurrent_switches"] = tuple(sc["recurrent_switches"])
        pop = dict(d["population"])
        pop["kappas"] = tuple(pop["kappas"])
        return cls(
            catalogue=CatalogueConfig(**d["catalogue"]),
            interaction=InteractionConfig(**d["interaction"]),
            scenario=ScenarioConfig(**sc),
            population=PopulationConfig(**pop),
        )

    def save_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load_json(cls, path: str | Path) -> "SimulationConfig":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


# Seed protocol (SIMULATION_DESIGN.md section 7). Evaluation seeds are never used for tuning.
TUNING_SEEDS: tuple[int, ...] = tuple(range(0, 5))
EVALUATION_SEEDS: tuple[int, ...] = tuple(range(100, 120))
