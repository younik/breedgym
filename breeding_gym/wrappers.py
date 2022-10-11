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
        f_index=yield_index,
        **kwargs
    ):
        if env is None:
            env = BreedingGym(**kwargs)
        super().__init__(env)

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
        try:
            n_offspring = ceil(self.individual_per_gen / n_crosses)
        except Exception as e:
            print(self.individual_per_gen)
            print(n_crosses)
            raise e
        if n_bests > self.individual_per_gen:
            raise ValueError("n_bests must be lower than individual_per_gen")
        if n_crosses > self.individual_per_gen:
            raise ValueError("n_crosses must be lower than individual_per_gen")

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


class ObserveStepWrapper(gym.Wrapper):

    def __init__(self, env):
        super().__init__(env)
        self.step_id = None
        self.observation_space = spaces.Discrete(BreedingGym.MAX_EPISODE_STEPS)

    def reset(self, **kwargs):
        _, info = super().reset(**kwargs)
        self.step_id = 0

        return self.step_id, info

    def step(self, action):
        _, reward, terminated, truncated, info = super().step(action)
        self.step_id += 1
        return self.step_id, reward, terminated, truncated, info
