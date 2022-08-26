import gym
import pandas as pd
import numpy as np


class BreedingGym(gym.Env):

    def __init__(self, chromosomes_map="data/map.txt", population="data/geno.txt", marker_effects="data/marker_effects.txt"):
        self.chromosomes_map = pd.read_table(chromosomes_map, sep="\t") #maybe groupby physics
        population_df = pd.read_table(population)
        self.population_names = population_df.index
        self.population = population_df.to_numpy()
        self.marker_effects = pd.read_table(marker_effects, sep="\t", index_col="Name")

        self.chr_set = self.chromosomes_map['CHR.PHYS'].unique()
        self.chr_indices = np.equal.outer(self.chromosomes_map['CHR.PHYS'].to_numpy(), self.chr_set)  # markers x n_chr
        chr_sizes = self.chromosomes_map.groupby('CHR.PHYS')['pred'].max()  # n_chr
        self.chr_sizes = chr_sizes.to_numpy()
        self.pred = self.chromosomes_map['pred'].to_numpy()

    def reset(self, seed=None, return_info=False, options=None):
        np.random.seed(seed)
        return self.observe()

    def step(self, action):
        """Action is a ndarray of shape n x 2, where n is the number of crosses. 
            Each row contains a couple of parent indices.
        """
        parents = self.population[action] # n x 2 x markers
        offspring = self._make_crosses(parents)
        self.population = offspring
        
        return self.observe()

    def _make_crosses(self, parents):
        breaking_points = np.random.rand(2, parents.shape[0], len(self.chr_set)) * self.chr_sizes.reshape(1, 1, -1)
        breaking_points[1, breaking_points[1] - breaking_points[0] < 20] = np.inf

        first_parent = np.random.randint(2, size=parents.shape[0])
        arange_parents = np.arange(parents.shape[0])
        progenies = parents[arange_parents, first_parent] # n x markers
        
        second_parent_mask = np.less.outer(breaking_points[0], self.pred) # n x n_chr x markers
        second_parent_mask = np.logical_and(second_parent_mask, np.less.outer(self.pred, breaking_points[1]))
        second_parent_mask = second_parent_mask[arange_parents, self.chr_indices.T]  # n x markers
        progenies[second_parent_mask] = parents[arange_parents, 1 - first_parent][second_parent_mask]

        return progenies

    def observe(self):
        return self.population, self.GEBV(), self.corrcoef()

    def GEBV(self):
        """
        Returns the estimated breeding values for each traits of interest of each individuals.
        If the population is composed by n individual, the output will be n x t, where t is the number of traits.
        """
        return np.dot(self.population, self.marker_effects)

    def corrcoef(self):
        return np.corrcoef(self.population)