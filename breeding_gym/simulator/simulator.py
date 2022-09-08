import numpy as np
import pandas as pd
from breeding_gym.utils.paths import DATA_PATH


class BreedingSimulator:

    def __init__(
        self,
        chromosomes_map=DATA_PATH.joinpath("map.txt"),
        marker_effects=DATA_PATH.joinpath("marker_effects.txt"),
    ):
        marker_effects_df = pd.read_table(
            marker_effects, sep="\t", index_col="Name"
        )
        self.trait_names = list(marker_effects_df.columns)
        self.marker_effects = marker_effects_df.to_numpy(np.float32)

        self.chr_map = pd.read_table(chromosomes_map, sep="\t")
        self.pred = self.chr_map['pred'].to_numpy()
        self.marker_chr, self.chr_set = self.chr_map['CHR.PHYS'].factorize()
        chr_phys_group = self.chr_map.groupby('CHR.PHYS')
        self.chr_sizes = chr_phys_group['pred'].max().to_numpy()

        self.recombination_vectors = []
        for chr_idx in range(len(self.chr_set)):
            marker_in_chr = np.count_nonzero(self.marker_chr == chr_idx)
            prob = 1.5 / marker_in_chr
            r = np.full(marker_in_chr, prob)
            r[0] = 0.5  # first element should be equally likely
            self.recombination_vectors.append(r)

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
        r = self.recombination_vectors[chr_idx]

        recombination_sites = np.random.binomial(
            1, r[None, :, None], size=(n_progenies, r.shape[0], 2)
        )
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
