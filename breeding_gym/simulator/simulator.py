from pathlib import Path
from typing import List, Optional, Union
import pandas as pd
from breeding_gym.simulator.gebv_model import GEBVModel
from breeding_gym.utils.paths import DATA_PATH
import numpy as np
import jax
import jax.numpy as jnp
from functools import partial
import random


@jax.jit
def _cross(parent, recombination_vec, random_key):
    samples = jax.random.uniform(random_key, shape=recombination_vec.shape)
    rec_sites = samples < recombination_vec
    crossover_mask =  jax.lax.associative_scan(jnp.logical_xor, rec_sites)

    crossover_mask = crossover_mask.astype(jnp.int8)
    progenies = jnp.take_along_axis(
        parent,
        crossover_mask[:, None],
        axis=-1
    )

    return progenies.squeeze()


class BreedingSimulator:

    def __init__(
        self,
        genetic_map: Union[Path, pd.DataFrame] = DATA_PATH.joinpath("genetic_map.txt"),
        trait_names: List["str"] = ["Yield"],
        h2: Optional[List[int]] = None,
        seed: Optional[int] = None
    ):
        if h2 is None:
            h2 = len(trait_names) * [1]
        assert len(h2) == len(trait_names)
        self.h2 = jnp.array(h2)
        self.trait_names = trait_names

        if isinstance(genetic_map, Path):
            types = {'Chr': 'int32', 'RecombRate': 'float32', "Effect": 'float32'}
            genetic_map = pd.read_table(genetic_map, sep="\t", dtype=types)

        mrk_effects = genetic_map["Effect"]
        self.GEBV_model = GEBVModel(
            marker_effects=mrk_effects.to_numpy()[:, None]
        )

        self.n_markers = len(genetic_map)
        self.recombination_vec = genetic_map["RecombRate"].to_numpy()

        # change semantic to "recombine now" instead of "recombine after"
        self.recombination_vec[1:] = self.recombination_vec[:-1]

        chr_map = genetic_map['Chr']
        first_mrk_map = np.zeros(len(chr_map), dtype='bool')
        first_mrk_map[1:] = chr_map[1:].values != chr_map[:-1].values
        first_mrk_map[0] = True
        self.recombination_vec[first_mrk_map] = 0.5  # first equally likely

        self.random_key = None
        if seed is None:
            seed = random.randint(0, 2**32)
        self.set_seed(seed)

    def set_seed(self, seed: int):
        self.random_key = jax.random.PRNGKey(seed)

    def load_population(self, file_name: Path):
        population = np.loadtxt(file_name, dtype='bool')
        return population.reshape(population.shape[0], self.n_markers, 2)

    def save_population(self, population: np.ndarray, file_name: Path):
        flatten_pop = population.reshape(population.shape[0], -1)
        np.savetxt(file_name, flatten_pop, fmt="%i")

    @partial(jax.vmap, in_axes=(None, 0))  # parallelize across individuals
    @partial(jax.vmap, in_axes=(None, 0), out_axes=1)  # parallelize across parents
    def cross(self, parent: jnp.ndarray):
        return _cross(parent, self.recombination_vec, self.random_key)

    def GEBV(self, population: jnp.ndarray) -> pd.DataFrame:
        GEBV = self.GEBV_model(population)
        return pd.DataFrame(GEBV, columns=self.trait_names)

    def phenotype(self, population: jnp.ndarray):
        noise = jax.random.normal(self.random_key, shape=(len(self.h2),))
        env_effect = (1 - self.h2) * (self.var_gebv * noise + self.mean_gebv)
        return self.h2 * self.GEBV(population) + env_effect

    def corrcoef(self, population: jnp.ndarray):
        monoploid_enc = population.reshape(population.shape[0], -1)
        mean_pop = jnp.mean(monoploid_enc, axis=0)
        pop_with_centroid = jnp.vstack([mean_pop, monoploid_enc])
        corrcoef = jnp.corrcoef(pop_with_centroid)
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
