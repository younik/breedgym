import gym
from gym import spaces
import pandas as pd
import numpy as np
from breeding_gym.paths import DATA_PATH


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
        )

        self.pred = self.chromosomes_map['pred'].to_numpy()
        chr_phys = self.chromosomes_map['CHR.PHYS']
        self.pred_chr, self.chr_set = chr_phys.factorize()
        self.chr_indices = np.equal.outer(chr_phys.to_numpy(), self.chr_set)
        chr_phys_group = self.chromosomes_map.groupby('CHR.PHYS')
        self.chr_sizes = chr_phys_group['pred'].max().to_numpy()

    def reset(self, seed=None, return_info=False, options=None):
        super().reset(seed=seed)
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
        return {"GEBV": self.GEBV(self.population)}

    def GEBV(self, population):
        """
        Returns the GEBV for each traits of each individuals.
        If the population is composed by n individual,
        the output will be n x t, where t is the number of traits.
        """
        return np.dot(population, self.marker_effects) / 2

    def corrcoef(self):
        return np.corrcoef(self.population)
