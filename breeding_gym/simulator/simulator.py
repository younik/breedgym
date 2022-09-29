import numpy as np
import jax.numpy as jnp
from jax import jit
import pandas as pd
from breeding_gym.utils.paths import DATA_PATH


@jit
def _cross(parents, marker_effects, crossover_mask):
    n_progenies = parents.shape[0]
    arange_prog = jnp.arange(n_progenies)[:, None]

    progenies = jnp.empty(
        shape=(n_progenies, len(marker_effects), 2),
        dtype='bool'
    )

    arange_markers = jnp.arange(len(marker_effects))[None, :]
    parent_0_mask = crossover_mask[:, :, 0].astype(jnp.int8)
    progenies.at[:, :, 0].set(parents[:, 0][
        arange_prog, arange_markers, parent_0_mask
    ])

    parent_1_mask = crossover_mask[:, :, 1].astype(jnp.int8)
    progenies.at[:, :, 1].set(parents[:, 1][
        arange_prog, arange_markers, parent_1_mask
    ])

    return progenies


class BreedingSimulator:

    def __init__(
        self,
        genetic_map=DATA_PATH.joinpath("genetic_map.txt"),
    ):
        genetic_map_df = pd.read_table(
            genetic_map, sep="\t", index_col="Marker"
        )
        self.trait_names = ["Yield"]
        self.marker_effects = genetic_map_df["Effect"].to_numpy(jnp.float32)
        positive_mask = self.marker_effects > 0
        self.max_gebvs = 2 * self.marker_effects[positive_mask].sum(axis=0)

        chr_map = genetic_map_df['Chr']
        self.marker_chr, self.chr_set = chr_map.factorize()

        self.recombination_vec = genetic_map_df["RecombRate"].to_numpy()
        
        # change semantic to "recombine now" instead of "recombine after"
        self.recombination_vec[1:] = self.recombination_vec[:-1]

        first_mrk_map = jnp.zeros(len(chr_map), dtype='bool')
        first_mrk_map.at[1:].set(chr_map[1:].values != chr_map[:-1].values)
        first_mrk_map.at[0].set(True)
        self.recombination_vec[first_mrk_map] = 0.5  # first equally likely


    def cross(self, parents):
        crossover_mask = self._get_crossover_mask(parents.shape[0])
        return _cross(parents, self.marker_effects, crossover_mask)

    def _get_crossover_mask(self, n_progenies):
        samples = np.random.rand(n_progenies, len(self.marker_effects), 2)
        recombination_sites = samples < self.recombination_vec[None, :, None]
        crossover_mask = np.logical_xor.accumulate(recombination_sites, axis=1)
        return crossover_mask


    def GEBV(self, population):
        monoploidy = population.sum(axis=-1)
        GEBV = jnp.dot(monoploidy, self.marker_effects)
        return pd.DataFrame(GEBV, columns=self.trait_names)

    def corrcoef(self, population):
        monoploid_enc = population.reshape(population.shape[0], -1)
        mean_pop = jnp.mean(monoploid_enc, axis=0)
        pop_with_centroid = jnp.vstack([mean_pop, monoploid_enc])
        corrcoef = jnp.corrcoef(pop_with_centroid)
        return corrcoef[0, 1:]
