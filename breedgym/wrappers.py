from math import ceil, sqrt
from typing import Callable, Optional, Tuple

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
from chromax.typing import Population
from gymnasium import spaces
from jaxtyping import Array, Float

from breedgym.breedgym import BreedGym
from breedgym.utils.index_functions import yield_index


class SimplifiedBreedGym(gym.Wrapper):

    metadata = BreedGym.metadata

    def __init__(
        self,
        env: Optional[BreedGym] = None,
        individual_per_gen: int = 2250,
        f_index: Optional[Callable[[Population["n"]], Float[Array, "n"]]] = None,
        **kwargs
    ):
        if env is None:
            env = BreedGym(**kwargs)
        super().__init__(env)

        if f_index is None:
            f_index = yield_index(env.simulator.GEBV_model)

        self.individual_per_gen = individual_per_gen

        self.observation_space = spaces.Dict(
            {
                "GEBV": spaces.Box(-1, 1, shape=(self.individual_per_gen,)),
                "corrcoef": spaces.Box(-1, 1, shape=(self.individual_per_gen,)),
            }
        )

        self.action_space = spaces.Dict(
            {
                "n_bests": spaces.Discrete(self.individual_per_gen - 1, start=2),
                "n_crosses": spaces.Discrete(self.individual_per_gen, start=1),
            }
        )

        self.f_index = f_index

    def reset(
        self, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[dict, dict]:
        if options is None:
            options = {}
        options["n_individuals"] = self.individual_per_gen
        if "index" in options.keys():
            self.f_index = options["index"]

        _, info = self.env.reset(seed=seed, options=options)

        return self._simplified_obs(), info

    def step(self, action: dict) -> Tuple[dict, float, bool, bool, dict]:
        n_bests = action["n_bests"]
        n_crosses = action["n_crosses"]
        n_offspring = ceil(self.individual_per_gen / n_crosses)

        if n_bests < 2:
            raise ValueError("n_bests must be higher or equal to 2")
        if n_crosses > self.individual_per_gen:
            raise ValueError("n_crosses must be lower or equal to individual_per_gen")

        self.unwrapped.population = self.simulator.select(
            population=self.env.population, k=n_bests, f_index=self.f_index
        )

        best_idx = np.arange(n_bests)
        diallel_indices = self.simulator._diallel_indices(best_idx)
        random_select_idx = self.np_random.choice(
            len(diallel_indices), n_crosses, replace=False
        )
        low_level_action = diallel_indices[random_select_idx]
        low_level_action = np.repeat(low_level_action, n_offspring, axis=0)
        low_level_action = low_level_action[: self.individual_per_gen]

        _, rew, terminated, truncated, info = self.env.step(low_level_action)
        obs = self._simplified_obs()
        return obs, rew, terminated, truncated, info

    @jax.jit
    def _correlation(population: Population["n"]) -> Float[Array, "n"]:
        monoploidy = jnp.sum(population, axis=-1) - 1
        mean_ind = jnp.mean(monoploidy, axis=0)
        norms = jnp.linalg.norm(monoploidy, axis=-1)
        norms *= jnp.linalg.norm(mean_ind, axis=-1)
        return jnp.dot(monoploidy, mean_ind) / norms

    def _simplified_obs(self) -> dict:
        norm_corr = SimplifiedBreedGym._correlation(self.population)
        norm_yield = self.GEBV["Yield"].to_numpy()
        return {"GEBV": norm_yield, "corrcoef": norm_corr}


class KBestBreedGym(SimplifiedBreedGym):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # max x, s.t. x * (x - 1) / 2 < individual_per_gen
        max_best = (1 + sqrt(1 + 8 * self.individual_per_gen)) // 2
        self.action_space = spaces.Discrete(int(max_best) - 1, start=2)

    def step(self, action: int) -> Tuple[dict, float, bool, bool, dict]:
        n_bests = action
        n_crosses = n_bests * (n_bests - 1) // 2

        return super().step({"n_bests": n_bests, "n_crosses": n_crosses})
