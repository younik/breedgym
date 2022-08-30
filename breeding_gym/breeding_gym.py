from typing import Any
import gym
from gym import spaces
import pandas as pd
import numpy as np


class BreedingGym(gym.Env):

    MAX_INDIVIDUALS = np.iinfo(np.int64).max
    
    def __init__(self, chromosomes_map="data/map.txt", population="data/geno.txt", marker_effects="data/marker_effects.txt"):
        self.observation_space = spaces.Sequence(
            spaces.Box(0, 1, shape=(19864,))
        )
        self.action_space = spaces.Sequence(
            spaces.Tuple((
                spaces.Discrete(self.MAX_INDIVIDUALS), 
                spaces.Discrete(self.MAX_INDIVIDUALS)
            ))
        )

        self.chromosomes_map = pd.read_table(chromosomes_map, sep="\t")
        population_df = pd.read_table(population, low_memory=False)
        self.population = population_df.to_numpy()
        self.population = (self.population * 2).astype(np.int8)
        self.germplasm = self.population
        self.marker_effects = pd.read_table(marker_effects, sep="\t", index_col="Name")

        self.pred = self.chromosomes_map['pred'].to_numpy()
        self.pred_chr, self.chr_set = self.chromosomes_map['CHR.PHYS'].factorize()
        self.chr_indices = np.equal.outer(self.chromosomes_map['CHR.PHYS'].to_numpy(), self.chr_set)  # markers x n_chr
        self.chr_sizes = self.chromosomes_map.groupby('CHR.PHYS')['pred'].max().to_numpy()  # n_chr

    def reset(self, seed=None, return_info=False, options=None):
        super().reset(seed=seed)
        self.population = self.germplasm
        if return_info:
            return self.population, self._get_info()
        else:
            return self.population

    def step(self, action):
        """Action is a ndarray of shape n x 2, where n is the number of crosses. 
            Each row contains a couple of parent indices.
        """
        parents = self.population[action]  # n x 2 x markers
        self.population = self._make_crosses(parents)
        
        info = self._get_info()
        return self.population, np.mean(info["GEBV"][:, 0]), False, False, info

    def _make_crosses(self, parents):
        breaking_points = np.random.rand(2, len(self.chr_set), parents.shape[0]) * self.chr_sizes[None, :, None]
        breaking_points[1, breaking_points[1] - breaking_points[0] < 20] = np.inf

        first_parent = np.random.randint(2, size=parents.shape[0])
        arange_parents = np.arange(parents.shape[0])
        progenies = parents[arange_parents, first_parent]  # n x markers

        second_parent_mask = np.logical_and(
                breaking_points[0, self.pred_chr] < self.pred[:, None], 
                breaking_points[1, self.pred_chr] > self.pred[:, None]
        ).T
        progenies[second_parent_mask] = parents[arange_parents, 1 - first_parent][second_parent_mask]

        return progenies

    def _get_info(self):
        return {"GEBV": self.GEBV(),
                "corrcoef": self.corrcoef()}

    def GEBV(self):
        """
        Returns the estimated breeding values for each traits of interest of each individuals.
        If the population is composed by n individual, the output will be n x t, where t is the number of traits.
        """
        return np.dot(self.population, self.marker_effects) / 2

    def corrcoef(self):
        return np.corrcoef(self.population)