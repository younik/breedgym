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
        return idx * np.ones((2, simulator.n_markers), dtype="bool")

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
