from pathlib import Path
from typing import Callable, List, Optional, Union
import pandas as pd
from breeding_gym.simulator.gebv_model import GEBVModel
from breeding_gym.utils.index_functions import yield_index
from breeding_gym.utils.paths import DATA_PATH
import numpy as np
import jax
import jax.numpy as jnp
from functools import partial
import random


GENETIC_MAP = DATA_PATH.joinpath("genetic_map.txt")


@jax.jit
def _cross(parent, recombination_vec, random_key):
    samples = jax.random.uniform(random_key, shape=recombination_vec.shape)
    rec_sites = samples < recombination_vec
    crossover_mask = jax.lax.associative_scan(jnp.logical_xor, rec_sites)

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
        genetic_map: Union[Path, pd.DataFrame] = GENETIC_MAP,
        trait_names: List["str"] = ["Yield"],
        h2: Optional[List[int]] = None,
        seed: Optional[int] = None,
        device=None
    ):
        if h2 is None:
            h2 = len(trait_names) * [1]
        assert len(h2) == len(trait_names)
        self.h2 = jnp.array(h2)
        self.trait_names = trait_names
        self.device = device

        if not isinstance(genetic_map, pd.DataFrame):
            types = {name: 'float32' for name in trait_names}
            types['RecombRate'] = 'float32'
            genetic_map = pd.read_table(genetic_map, sep="\t", dtype=types)

        mrk_effects = genetic_map[trait_names]
        self.GEBV_model = GEBVModel(
            marker_effects=mrk_effects.to_numpy(),
            device=self.device
        )

        self.n_markers = len(genetic_map)
        recombination_vec = genetic_map["RecombRate"].to_numpy()

        # change semantic to "recombine now" instead of "recombine after"
        recombination_vec[1:] = recombination_vec[:-1]

        chr_map = genetic_map['CHR.PHYS']
        first_mrk_map = np.zeros(len(chr_map), dtype='bool')
        first_mrk_map[1:] = chr_map.iloc[1:].values != chr_map.iloc[:-1].values
        first_mrk_map[0] = True
        recombination_vec[first_mrk_map] = 0.5  # first equally likely
        self.recombination_vec = jax.device_put(
            recombination_vec,
            device=self.device
        )

        self.random_key = None
        if seed is None:
            seed = random.randint(0, 2**32)
        self.set_seed(seed)

    def set_seed(self, seed: int):
        self.random_key = jax.random.PRNGKey(seed)

    def load_population(self, file_name: Path):
        population = np.loadtxt(file_name, dtype='bool')
        population = population.reshape(population.shape[0], self.n_markers, 2)
        return jax.device_put(population, device=self.device)

    def save_population(self, population: np.ndarray, file_name: Path):
        flatten_pop = population.reshape(population.shape[0], -1)
        np.savetxt(file_name, flatten_pop, fmt="%i")

    def cross(self, parents: np.ndarray):
        keys = jax.random.split(self.random_key, num=len(parents) * 2 + 1)
        self.random_key = keys[0]
        split_keys = keys[1:].reshape(len(parents), 2, 2)
        return self._vmap_cross(parents, split_keys)

    @partial(jax.vmap, in_axes=(None, 0, 0))  # parallelize across individuals
    @partial(jax.vmap, in_axes=(None, 0, 0), out_axes=1)  # parallelize parents
    def _vmap_cross(self, parent: np.ndarray, random_key: jax.random.PRNGKey):
        return _cross(parent, self.recombination_vec, random_key)

    def double_haploid(self, population: np.ndarray):
        keys = jax.random.split(self.random_key, num=len(population) + 1)
        self.random_key = keys[0]
        split_keys = keys[1:]
        return self._vmap_dh(population, split_keys)

    @partial(jax.vmap, in_axes=(None, 0, 0))  # parallelize across individuals
    def _vmap_dh(self, population: np.ndarray, random_key: jax.random.PRNGKey):
        haploid = _cross(population, self.recombination_vec, random_key)
        return jnp.broadcast_to(haploid[:, None], shape=(*haploid.shape, 2))

    def diallel(self, population: np.ndarray, n_offspring: int = 1):
        if n_offspring < 1:
            raise ValueError("n_offspring must be higher or equal to 1")

        all_indices = np.arange(len(population))
        diallel_indices = self._diallel_indices(all_indices)
        cross_indices = np.repeat(diallel_indices, n_offspring, axis=0)
        return self.cross(population[cross_indices])

    def _diallel_indices(self, indices):
        mesh1, mesh2 = jnp.meshgrid(indices, indices)
        triu_indices = jnp.triu_indices(len(indices), k=1)
        mesh1 = mesh1[triu_indices]
        mesh2 = mesh2[triu_indices]
        return jnp.stack([mesh1, mesh2], axis=1)

    def random_crosses(
        self,
        population: np.ndarray,
        n_crosses: int,
        n_offspring: int = 1
    ):
        if n_crosses < 1:
            raise ValueError("n_crosses must be higher or equal to 1")
        if n_offspring < 1:
            raise ValueError("n_offspring must be higher or equal to 1")

        all_indices = np.arange(len(population))
        diallel_indices = self._diallel_indices(all_indices)
        random_select_idx = np.random.choice(
            len(diallel_indices), n_crosses, replace=False
        )
        cross_indices = diallel_indices[random_select_idx]

        cross_indices = np.repeat(cross_indices, n_offspring, axis=0)
        return self.cross(population[cross_indices])

    def select(
        self,
        population: np.ndarray,
        k: int,
        f_index: Callable[[np.ndarray], int] = None
    ):
        if f_index is None:
            f_index = yield_index(self.GEBV_model)

        indices = f_index(population)
        _, best_pop = jax.lax.top_k(indices, k)
        return population[best_pop, :, :]

    def GEBV(self, population: np.ndarray) -> pd.DataFrame:
        GEBV = self.GEBV_model(population)
        return pd.DataFrame(GEBV, columns=self.trait_names)

    def phenotype(self, population: np.ndarray):
        noise = jax.random.normal(self.random_key, shape=(len(self.h2),))
        env_effect = (1 - self.h2) * (self.var_gebv * noise + self.mean_gebv)
        return self.h2 * self.GEBV(population) + env_effect

    def corrcoef(self, population: np.ndarray):
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
