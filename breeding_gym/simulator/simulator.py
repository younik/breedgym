import numpy as np
import pandas as pd
from breeding_gym.utils.paths import DATA_PATH


class BreedingSimulator:

    def __init__(
        self,
        genetic_map=DATA_PATH.joinpath("genetic_map.txt"),
    ):
        genetic_map_df = pd.read_table(
            genetic_map, sep="\t", index_col="Marker"
        )
        self.trait_names = ["Yield"]
        self.marker_effects = genetic_map_df["Effect"].to_numpy(np.float32)
        positive_mask = self.marker_effects > 0
        self.max_gebvs = 2 * self.marker_effects[positive_mask].sum(axis=0)

        chr_map = genetic_map_df['Chr']
        self.marker_chr, self.chr_set = chr_map.factorize()

        self.recombination_vec = genetic_map_df["RecombRate"].to_numpy()
        
        # change semantic to "recombine now" instead of "recombine after"
        self.recombination_vec[1:] = self.recombination_vec[:-1]

        first_mrk_map = np.zeros(len(chr_map), dtype='bool')
        first_mrk_map[1:] = chr_map[1:].values != chr_map[:-1].values
        first_mrk_map[0] = True
        self.recombination_vec[first_mrk_map] = 0.5  # first equally likely

    def cross(self, parents):
        n_progenies = parents.shape[0]
        arange_prog = np.arange(n_progenies)[:, None]

        progenies = np.empty(
            shape=(n_progenies, len(self.marker_effects), 2),
            dtype='bool'
        )

        crossover_mask = self._get_crossover_mask(n_progenies)

        arange_markers = np.arange(len(self.marker_effects))[None, :]
        parent_0_mask = crossover_mask[:, :, 0].astype(np.int8)
        progenies[:, :, 0] = parents[:, 0][
            arange_prog, arange_markers, parent_0_mask
        ]

        parent_1_mask = crossover_mask[:, :, 1].astype(np.int8)
        progenies[:, :, 1] = parents[:, 1][
            arange_prog, arange_markers, parent_1_mask
        ]

        return progenies

    def _get_crossover_mask(self, n_progenies):
        samples = np.random.rand(n_progenies, len(self.marker_effects), 2)
        recombination_sites = samples < self.recombination_vec[None, :, None]
        crossover_mask = np.logical_xor.accumulate(recombination_sites, axis=1)
        return crossover_mask

    def GEBV(self, population):
        monoploidy = population.sum(axis=-1, dtype=np.float32)
        GEBV = np.dot(monoploidy, self.marker_effects)
        return pd.DataFrame(GEBV, columns=self.trait_names)

    def corrcoef(self, population):
        monoploid_enc = population.reshape(population.shape[0], -1)
        mean_pop = np.mean(monoploid_enc, axis=0, dtype=np.float32)
        pop_with_centroid = np.vstack([mean_pop, monoploid_enc])
        corrcoef = np.corrcoef(pop_with_centroid, dtype=np.float32)
        return corrcoef[0, 1:]
