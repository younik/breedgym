from pathlib import Path
import numpy as np
import pandas as pd
from breeding_gym.simulator.gebv_model import GEBVModel
from breeding_gym.utils.paths import DATA_PATH


def _cross(parents, crossover_mask):
    crossover_mask = crossover_mask.astype(np.int8)
    progenies = np.take_along_axis(
        parents,
        crossover_mask[:, :, :, None],
        axis=-1
    )

    return progenies.squeeze(-1).transpose(0, 2, 1)


class BreedingSimulator:

    def __init__(
        self,
        genetic_map: Path = DATA_PATH.joinpath("genetic_map.txt"),
        trait_names: list[str] = ["Yield"],
        h2: list[int] | None = None
    ):
        if h2 is None:
            h2 = len(trait_names) * [1]
        assert len(h2) == len(trait_names)
        self.h2 = np.array(h2)
        self.trait_names = trait_names

        genetic_map_df = pd.read_table(
            genetic_map, sep="\t", index_col="Marker"
        )

        mrk_effects = genetic_map_df["Effect"]
        self.GEBV_model = GEBVModel(
            marker_effects=mrk_effects.to_numpy(np.float32)[:, None]
        )

        self.n_markers = len(genetic_map_df)
        chr_map = genetic_map_df['Chr']
        self.marker_chr, self.chr_set = chr_map.factorize()

        self.recombination_vec = genetic_map_df["RecombRate"].to_numpy()

        # change semantic to "recombine now" instead of "recombine after"
        self.recombination_vec[1:] = self.recombination_vec[:-1]

        first_mrk_map = np.zeros(len(chr_map), dtype='bool')
        first_mrk_map[1:] = chr_map[1:].values != chr_map[:-1].values
        first_mrk_map[0] = True
        self.recombination_vec[first_mrk_map] = 0.5  # first equally likely

    def cross(self, parents: np.ndarray):
        crossover_mask = self._get_crossover_mask(parents.shape[0])
        return _cross(parents, crossover_mask)

    def _get_crossover_mask(self, n_progenies):
        samples = np.random.rand(n_progenies, 2, self.n_markers)
        recombination_sites = samples < self.recombination_vec[None, None, :]
        crossover_mask = np.logical_xor.accumulate(recombination_sites, axis=2)
        return crossover_mask

    def GEBV(self, population: np.ndarray) -> pd.DataFrame:
        GEBV = self.GEBV_model(population)
        return pd.DataFrame(GEBV, columns=self.trait_names)

    def phenotype(self, population: np.ndarray):
        env_effect = (1 - self.h2) * self.var_gebv * \
            np.random.randn(len(self.h2))
        return self.h2 * self.GEBV(population) + env_effect

    def corrcoef(self, population: np.ndarray):
        monoploid_enc = population.reshape(population.shape[0], -1)
        mean_pop = np.mean(monoploid_enc, axis=0, dtype=np.float32)
        pop_with_centroid = np.vstack([mean_pop, monoploid_enc])
        corrcoef = np.corrcoef(pop_with_centroid, dtype=np.float32)
        return corrcoef[0, 1:]

    @property
    def max_gebv(self):
        return self.GEBV_model.max

    @property
    def min_gebv(self):
        return self.GEBV_model.min

    @property
    def mean_gebv(self):
        return self.GEBV_model.mean

    @property
    def var_gebv(self):
        return self.GEBV_model.var
