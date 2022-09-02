from math import sqrt, ceil
import gym
from gym import spaces
import pandas as pd
import numpy as np
from breeding_gym.utils.paths import DATA_PATH


class BreedingGym(gym.Env):

    MAX_INDIVIDUALS = np.iinfo(np.int64).max

    def __init__(
        self,
        chromosomes_map=DATA_PATH.joinpath("map.txt"),
        initial_population=DATA_PATH.joinpath("geno.txt"),
        marker_effects=DATA_PATH.joinpath("marker_effects.txt")
    ):
        self.observation_space = spaces.Sequence(
            spaces.Box(0, 1, shape=(19864,))
        )
        self.action_space = spaces.Sequence(
            spaces.Tuple((
                spaces.Discrete(self.MAX_INDIVIDUALS),
                spaces.Discrete(self.MAX_INDIVIDUALS)
            ))
        )

        population_df = pd.read_table(initial_population, low_memory=False)
        self.germplasm = population_df.to_numpy()
        self.germplasm = (self.germplasm * 2).astype(np.int8)
        self.population = None

        self.chromosomes_map = pd.read_table(chromosomes_map, sep="\t")
        self.marker_effects = pd.read_table(
            marker_effects, sep="\t", index_col="Name"
        ).to_numpy(np.float32)

        self.pred = self.chromosomes_map['pred'].to_numpy()
        chr_phys = self.chromosomes_map['CHR.PHYS']
        self.pred_chr, self.chr_set = chr_phys.factorize()
        self.chr_indices = np.equal.outer(chr_phys.to_numpy(), self.chr_set)
        chr_phys_group = self.chromosomes_map.groupby('CHR.PHYS')
        self.chr_sizes = chr_phys_group['pred'].max().to_numpy()

    def reset(self, seed=None, return_info=False, options=None):
        super().reset(seed=seed)

        if options is not None and "n_individuals" in options.keys():
            selected_indices = np.random.choice(
                len(self.germplasm),
                options["n_individuals"],
                replace=False
            )
            self.population = self.germplasm[selected_indices]
        else:
            self.population = self.germplasm

        if return_info:
            return self.population, self._get_info()
        else:
            return self.population

    def step(self, action):
        """Action is an array of shape n x 2, where n is the number of crosses.
           Each row contains a couple of parent indices.
        """
        parents = self.population[action]  # n x 2 x markers
        self.population = self._make_crosses(parents)

        info = self._get_info()
        return self.population, np.mean(info["GEBV"][:, 0]), False, False, info

    def _make_crosses(self, parents):
        n_progenies = parents.shape[0]

        breaking_points = np.random.rand(2, len(self.chr_set), n_progenies)
        breaking_points *= self.chr_sizes[None, :, None]
        invalid_bp_mask = breaking_points[1] - breaking_points[0] < 20
        breaking_points[1, invalid_bp_mask] = np.inf

        first_parent_idx = np.random.randint(2, size=n_progenies)
        arange_parents = np.arange(n_progenies)
        progenies = parents[arange_parents, first_parent_idx]  # n x markers

        second_parent_mask = np.logical_and(
            breaking_points[0, self.pred_chr] < self.pred[:, None],
            breaking_points[1, self.pred_chr] > self.pred[:, None]
        ).T
        second_parent_mask = np.ascontiguousarray(second_parent_mask)
        second_parents = parents[arange_parents, 1 - first_parent_idx]
        progenies[second_parent_mask] = second_parents[second_parent_mask]

        return progenies

    def _get_info(self):
        return {"GEBV": self.GEBV}

    @property
    def GEBV(self):
        """
        Returns the GEBV for each traits of each individuals.
        If the population is composed by n individual,
        the output will be n x t, where t is the number of traits.
        """
        return np.dot(self.population, self.marker_effects) / 2

    def corrcoef(self):
        return np.corrcoef(self.population)


def default_f_index(GEBV):
    GEBV_copy = np.copy(GEBV)
    GEBV_copy[:, 2] = np.abs(GEBV_copy[:, 2])

    weights = np.array([2, -1, 1, 1, 1])
    return np.dot(GEBV_copy, weights)


class SimplifiedBreedingGym(gym.Wrapper):

    def __init__(
        self,
        env=None,
        individual_per_gen=2250,
        f_index=default_f_index
    ):
        if env is None:
            env = BreedingGym()

        super().__init__(env)
        self.individual_per_gen = individual_per_gen

        self.observation_space = spaces.Dict({
            "GEBV": spaces.Box(-15, 15, shape=(self.individual_per_gen,)),
            "corrcoef": spaces.Box(-0.5, 0.5, shape=(self.individual_per_gen,))
        })

        # max x | x * (x - 1) / 2 < individual_per_gen
        max_best = (1 + sqrt(1 + 8 * self.individual_per_gen)) // 2
        self.action_space = spaces.Discrete(int(max_best - 1), start=2)

        self.f_index = f_index

    def reset(self, seed=None, return_info=True, options=None):
        assert return_info is True
        if options is None:
            options = {}
        options["n_individuals"] = self.individual_per_gen

        pop, info = self.env.reset(seed, return_info, options)

        return self._simplified_obs(pop, info), info

    def step(self, action):
        children = action * (action - 1) / 2
        n_offspring = ceil(self.individual_per_gen / children)

        indices = self.f_index(self.GEBV)

        # retrieve the `action` best population indices
        best_pop = np.argpartition(indices, -action)[-action:]

        mesh1, mesh2 = np.meshgrid(best_pop, best_pop)
        triu_indices = np.triu_indices(action, k=1)
        mesh1 = mesh1[triu_indices]
        mesh2 = mesh2[triu_indices]
        low_level_action = np.stack([mesh1, mesh2], axis=1)
        low_level_action = np.repeat(low_level_action, n_offspring, axis=0)
        low_level_action = low_level_action[:self.individual_per_gen]

        pop, _, terminated, truncated, info = self.env.step(low_level_action)
        obs = self._simplified_obs(pop, info)
        return obs, np.mean(obs["GEBV"]), terminated, truncated, info

    def _simplified_obs(self, pop, info):
        mean_pop = np.mean(pop, axis=0, dtype=np.float32)
        corrcoef = np.corrcoef(np.vstack([mean_pop, pop]), dtype=np.float32)
        corrcoef -= 0.5
        clipped_GEBV = np.clip(info["GEBV"][:, 0], 0, 30) - 15
        return {"GEBV": clipped_GEBV, "corrcoef": corrcoef[1:, 0]}
