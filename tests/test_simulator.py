from breeding_gym.simulator import BreedingSimulator
import pytest
import numpy as np
from breeding_gym.utils.paths import DATA_PATH


@pytest.mark.parametrize("idx", [0, 1])
def test_cross_r(idx):
    simulator = BreedingSimulator(
        genetic_map=DATA_PATH.joinpath("small_genetic_map.txt"),
    )

    def const_co_mask():
        return idx * np.ones(simulator.n_markers, dtype="bool")

    simulator._get_crossover_mask = const_co_mask
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
    simulator = BreedingSimulator(
        genetic_map=DATA_PATH.joinpath("small_genetic_map.txt")
    )

    parent_0 = np.array([
        [0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0],
        [1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1]
    ], dtype='bool')
    parent_1 = np.array([
        [0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0],
        [0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0]
    ], dtype='bool')
    assert parent_0.shape == parent_1.shape

    co_mask = np.array(
        [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0],
        dtype=np.int8
    )
    assert len(co_mask) == parent_1.shape[1]

    simulator._get_crossover_mask = lambda : co_mask
    parents = np.empty((1, 2, *parent_0.T.shape), dtype='bool')
    parents[0, 0] = parent_0.T
    parents[0, 1] = parent_1.T
    child = simulator.cross(parents)

    assert child.shape == (1, *parent_0.T.shape)

    for idx in range(len(co_mask)):
        assert child[1, idx, 0] == parent_0[co_mask[idx], idx]
        assert child[1, idx, 1] == parent_1[co_mask[idx], idx]

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
