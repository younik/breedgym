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

        self.marker_chr, self.chr_set = genetic_map_df['Chr'].factorize()

        self.r_vectors = genetic_map_df.groupby("Chr")["RecombRate"].agg(
            lambda chr_r: chr_r.to_list()
        ).values
        for r in self.r_vectors:
            # change semantic to "recombine now" instead of "recombine after"
            r[1:] = r[:-1]
            r[0] = 0.5  # first index equally likely

    def cross(self, parents):
        n_progenies = parents.shape[0]
        arange_prog = np.arange(n_progenies).reshape(-1, 1)

        progenies = np.empty(
            shape=(n_progenies, len(self.marker_effects), 2),
            dtype='bool'
        )
        for chr_idx in range(len(self.chr_set)):
            crossover_mask = self._get_crossover_mask(n_progenies, chr_idx)
            marker_mask = self.marker_chr == chr_idx

            parent_0 = parents[:, 0, marker_mask]
            arange_markers = np.arange(parent_0.shape[1]).reshape(1, -1)
            progenies[:, marker_mask, 0] = parent_0[
                arange_prog, arange_markers, crossover_mask[:, :, 0]
            ]

            parent_1 = parents[:, 1, marker_mask]
            progenies[:, marker_mask, 1] = parent_1[
                arange_prog, arange_markers, crossover_mask[:, :, 1]
            ]

        return progenies

    def _get_crossover_mask(self, n_progenies, chr_idx):
        r = self.r_vectors[chr_idx]
        samples = np.random.rand(n_progenies, r.shape[0], 2)
        recombination_sites = samples < r[None, :, None]
        crossover_mask = np.cumsum(recombination_sites, axis=1) % 2
        return crossover_mask

    def GEBV(self, population):
        monoploidy = population.mean(axis=-1, dtype=np.float32)
        GEBV = np.dot(monoploidy, self.marker_effects)
        return pd.DataFrame(GEBV, columns=self.trait_names)

    def corrcoef(self, population):
        monoploid_enc = population.sum(axis=-1)
        mean_pop = np.mean(monoploid_enc, axis=0, dtype=np.float32)
        pop_with_centroid = np.vstack([mean_pop, monoploid_enc])
        corrcoef = np.corrcoef(pop_with_centroid, dtype=np.float32)
        return corrcoef[0, 1:]
