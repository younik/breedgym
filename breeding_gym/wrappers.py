from math import ceil
from breeding_gym.breeding_gym import BreedingGym
from breeding_gym.utils.index_functions import yield_index
import gym
from gym import spaces
import numpy as np


class SimplifiedBreedingGym(gym.Wrapper):

    metadata = BreedingGym.metadata

    def __init__(
        self,
        individual_per_gen=2250,
        f_index=yield_index,
        **kwargs
    ):
        env = BreedingGym(**kwargs)
        super().__init__(env)

        self.individual_per_gen = individual_per_gen

        self.observation_space = spaces.Dict({
            "GEBV": spaces.Box(-1, 1, shape=(self.individual_per_gen,)),
            "corrcoef": spaces.Box(-0.5, 0.5, shape=(self.individual_per_gen,))
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

        _, info = self.env.reset(seed, options)

        return self._simplified_obs(info), info

    def step(self, action):
        n_bests = action["n_bests"]
        n_crosses = action["n_crosses"]
        n_offspring = ceil(self.individual_per_gen / n_crosses)

        indices = self.f_index(self)

        # retrieve the `n_bests` best population indices
        best_pop = np.argpartition(indices, -n_bests)[-n_bests:]

        mesh1, mesh2 = np.meshgrid(best_pop, best_pop)
        triu_indices = np.triu_indices(n_bests, k=1)
        mesh1 = mesh1[triu_indices]
        mesh2 = mesh2[triu_indices]
        low_level_action = np.stack([mesh1, mesh2], axis=1)

        random_select_idx = np.random.choice(
            len(low_level_action), n_crosses, replace=False
        )
        low_level_action = low_level_action[random_select_idx]

        low_level_action = np.repeat(low_level_action, n_offspring, axis=0)
        low_level_action = low_level_action[:self.individual_per_gen]

        _, rew, terminated, truncated, info = self.env.step(low_level_action)
        obs = self._simplified_obs(info)
        return obs, rew, terminated, truncated, info

    def _simplified_obs(self, info):
        norm_corrcoef = self.env.corrcoef * 2 - 1
        norm_GEBV = info["GEBV"]["Yield"].to_numpy() - self.simulator.mean_gebv
        norm_GEBV /= self.simulator.max_gebv - self.simulator.min_gebv
        return {"GEBV": norm_GEBV, "corrcoef": norm_corrcoef}


class KBestBreedingGym(SimplifiedBreedingGym):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.action_space = spaces.Discrete(
            self.individual_per_gen - 1,
            start=2
        )

    def step(self, action):
        n_bests = action
        n_crosses = n_bests * (n_bests - 1) // 2

        return super().step({"n_bests": n_bests, "n_crosses": n_crosses})
