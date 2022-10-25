from breeding_gym.simulator import BreedingSimulator
import pytest
import numpy as np
from breeding_gym.utils.paths import DATA_PATH
import pandas as pd


class MockSimulator(BreedingSimulator):

    def __init__(
        self,
        n_markers=None,
        marker_effects=None,
        recombination_vec=None,
        n_chr=10,
        **kwargs
    ):
        self.n_markers = n_markers
        if self.n_markers is None:
            if marker_effects is not None:
                self.n_markers = len(marker_effects)
            elif recombination_vec is not None:
                self.n_markers = len(recombination_vec)
            else:
                raise Exception(
                    "You must specify at least one between ",
                    "n_markers, marker_effects and recombination_vec"
                )

        if marker_effects is None:
            marker_effects = np.random.randn(self.n_markers)
        if recombination_vec is None:
            recombination_vec = np.random.uniform(size=self.n_markers)
            recombination_vec /= self.n_markers / 20

        if len(marker_effects) != self.n_markers or \
           len(recombination_vec) != self.n_markers:
            raise Exception(
                "Incompatible arguments. ",
                f"Lenght of marker_effects is {len(marker_effects)}.",
                f"Lenght of recombination_vec is {len(recombination_vec)}."

            )

        chromosomes = np.arange(self.n_markers) // (self.n_markers // n_chr)

        data = np.vstack([chromosomes, recombination_vec, marker_effects]).T
        genetic_map = pd.DataFrame(
            data, columns=["Chr", "RecombRate", "Effect"])

        super().__init__(genetic_map=genetic_map, **kwargs)
        self.recombination_vec = recombination_vec

    def load_population(self, n_individual=100):
        return np.random.choice(
            a=[False, True],
            size=(n_individual, self.n_markers, 2),
            p=[0.5, 0.5]
        )


@pytest.mark.parametrize("idx", [0, 1])
def test_cross_r(idx):
    n_markers = 1000
    recombination_vec = np.zeros(n_markers, dtype="bool")
    recombination_vec[0] = idx
    simulator = MockSimulator(recombination_vec=recombination_vec)

    size = (1, 2, simulator.n_markers, 2)
    parents = np.random.choice(a=[False, True], size=size, p=[0.5, 0.5])

    new_pop = simulator.cross(parents)

    assert new_pop.shape == (1, simulator.n_markers, 2)

    ind = new_pop[0]
    assert np.all(ind[:, 0] == parents[0, 0, :, idx])
    assert np.all(ind[:, 1] == parents[0, 1, :, idx])


def test_equal_parents():
    simulator = BreedingSimulator(
        genetic_map=DATA_PATH.joinpath("small_genetic_map.txt")
    )

    parents = np.zeros((1, 2, simulator.n_markers, 2), dtype="bool")
    child = simulator.cross(parents)
    assert np.all(child == 0)

    parents = np.ones((1, 2, simulator.n_markers, 2), dtype="bool")
    child = simulator.cross(parents)
    assert np.all(child == 1)


def test_ad_hoc_cross():
    parent_0 = np.array([
        [0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0],
        [1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1]
    ], dtype='bool')
    parent_1 = np.array([
        [0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0],
        [0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0]
    ], dtype='bool')
    assert parent_0.shape == parent_1.shape

    rec_vec = np.array(
        [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
        dtype=np.int8
    )
    assert len(rec_vec) == parent_1.shape[1]
    simulator = MockSimulator(recombination_vec=rec_vec)

    parents = np.empty((1, 2, *parent_0.T.shape), dtype='bool')
    parents[0, 0] = parent_0.T
    parents[0, 1] = parent_1.T
    child = simulator.cross(parents)

    assert child.shape == (1, *parent_0.T.shape)

    chr_idx = 0
    for mrk_idx, rec_prob in enumerate(rec_vec):
        if rec_prob == 1:
            chr_idx = 1 - chr_idx
        assert child[1, mrk_idx, 0] == parent_0[chr_idx, mrk_idx]
        assert child[1, mrk_idx, 1] == parent_1[chr_idx, mrk_idx]


def test_cross_two_times():
    n_markers = 100_000
    n_ind = 2
    simulator = MockSimulator(n_markers=n_markers)
    population = simulator.load_population(n_ind)

    parents = population[np.array([[0, 1], [0, 1]])]
    children = simulator.cross(parents)

    assert np.any(children[0] != children[1])


def test_double_haploid():
    n_markers = 1000
    n_ind = 100

    rec_vector = np.zeros(n_markers, dtype='bool')
    rec_vector[0] = 1
    simulator = MockSimulator(recombination_vec=rec_vector)
    population = simulator.load_population(n_ind)

    new_pop = simulator.double_haploid(population)

    assert new_pop.shape == population.shape
    assert np.all(new_pop[:, :, 0] == population[:, :, 1])
    assert np.all(new_pop[:, :, 1] == population[:, :, 1])


def test_diallel():
    n_markers = 1000
    n_ind = 100
    simulator = MockSimulator(n_markers=n_markers)
    population = simulator.load_population(n_ind)

    new_pop = simulator.diallel(population)
    assert len(new_pop) == n_ind * (n_ind - 1) // 2

    new_pop = simulator.diallel(population, n_offspring=10)
    assert len(new_pop) == n_ind * (n_ind - 1) // 2 * 10


def test_random_crosses():
    n_markers = 1000
    n_ind = 100
    simulator = MockSimulator(n_markers=n_markers)
    population = simulator.load_population(n_ind)

    n_crosses = 300
    new_pop = simulator.random_crosses(population, n_crosses=n_crosses)
    assert new_pop.shape == (n_crosses, n_markers, 2)

    n_offspring = 10
    new_pop = simulator.random_crosses(
        population=population,
        n_crosses=n_crosses,
        n_offspring=n_offspring
    )
    assert new_pop.shape == (n_crosses * n_offspring, n_markers, 2)


def test_phenotyping():
    simulator = BreedingSimulator(
        genetic_map=DATA_PATH.joinpath("small_genetic_map.txt"),
        h2=[0.5]
    )

    size = (10, simulator.n_markers, 2)
    pop = np.random.choice(a=[False, True], size=size, p=[0.5, 0.5])

    assert np.any(simulator.phenotype(pop) != simulator.GEBV(pop))

    simulator.h2 = np.array([1])
    assert np.all(simulator.phenotype(pop) == simulator.GEBV(pop))
