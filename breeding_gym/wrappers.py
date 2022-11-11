from math import ceil, sqrt
from breeding_gym.breeding_gym import BreedingGym
from breeding_gym.utils.index_functions import yield_index
import gym
from gym import spaces
import numpy as np


class SimplifiedBreedingGym(gym.Wrapper):

    metadata = BreedingGym.metadata

    def __init__(
        self,
        env=None,
        individual_per_gen=2250,
        f_index=None,
        **kwargs
    ):
        if env is None:
            env = BreedingGym(**kwargs)
        super().__init__(env)

        if f_index is None:
            f_index = yield_index(env.simulator.GEBV_model)

        self.individual_per_gen = individual_per_gen

        self.observation_space = spaces.Dict({
            "GEBV": spaces.Box(-1, 1, shape=(self.individual_per_gen,)),
            "corrcoef": spaces.Box(-1, 1, shape=(self.individual_per_gen,))
        })

        self.action_space = spaces.Dict({
            "n_bests": spaces.Discrete(self.individual_per_gen - 1, start=2),
            "n_crosses": spaces.Discrete(self.individual_per_gen, start=1)
        })

        self.f_index = f_index

    def reset(self, seed=None, options=None):
        if options is None:
            options = {}
        options["n_individuals"] = self.individual_per_gen
        if "index" in options.keys():
            self.f_index = options["index"]

        _, info = self.env.reset(seed=seed, options=options)

        return self._simplified_obs(), info

    def step(self, action):
        n_bests = action["n_bests"]
        n_crosses = action["n_crosses"]
        n_offspring = ceil(self.individual_per_gen / n_crosses)

        if n_bests < 2:
            raise ValueError("n_bests must be higher or equal to 2")
        if n_crosses > self.individual_per_gen:
            raise ValueError(
                "n_crosses must be lower or equal to individual_per_gen"
            )

        self.unwrapped.population = self.simulator.select(
            population=self.env.population,
            k=n_bests,
            f_index=self.f_index
        )

        best_idx = np.arange(n_bests)
        diallel_indices = self.simulator._diallel_indices(best_idx)
        random_select_idx = np.random.choice(
            len(diallel_indices), n_crosses, replace=False
        )
        low_level_action = diallel_indices[random_select_idx]
        low_level_action = np.repeat(low_level_action, n_offspring, axis=0)
        low_level_action = low_level_action[:self.individual_per_gen]

        _, rew, terminated, truncated, info = self.env.step(low_level_action)
        obs = self._simplified_obs()
        return obs, rew, terminated, truncated, info

    def _simplified_obs(self):
        norm_corrcoef = self.corrcoef * 2 - 1
        return {
            "GEBV": self.norm_GEBV["Yield"].to_numpy(),
            "corrcoef": norm_corrcoef
        }


class KBestBreedingGym(SimplifiedBreedingGym):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # max x, s.t. x * (x - 1) / 2 < individual_per_gen
        max_best = (1 + sqrt(1 + 8 * self.individual_per_gen)) // 2
        self.action_space = spaces.Discrete(int(max_best) - 1, start=2)

    def step(self, action):
        n_bests = action
        n_crosses = n_bests * (n_bests - 1) // 2

        return super().step({"n_bests": n_bests, "n_crosses": n_crosses})
