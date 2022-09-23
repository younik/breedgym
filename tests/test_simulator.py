from breeding_gym.simulator.simulator import BreedingSimulator
import pytest
import numpy as np
from breeding_gym.utils.paths import DATA_PATH
import gym


@pytest.mark.parametrize("idx", [0, 1])
def test_cross_r(idx):
    def const_co_mask(self, n_progenies):
        n_markers = len(self.marker_effects)
        return idx * np.ones((n_progenies, n_markers, 2), dtype="bool")
    BreedingSimulator._get_crossover_mask = const_co_mask

    env = gym.make("BreedingGym",
                   initial_population=DATA_PATH.joinpath("small_geno.txt"),
                   genetic_map=DATA_PATH.joinpath("small_genetic_map.txt"),
                   )
    init_pop = env.reset()
    p0, p1 = init_pop[0], init_pop[1]
    assert p0.shape == p1.shape

    new_pop, _, _, _ = env.step(np.array([[0, 1]]))

    assert new_pop.shape == (1, p0.shape[0], 2)

    ind = new_pop[0]
    assert np.all(ind[:, 0] == p0[:, idx])
    assert np.all(ind[:, 1] == p1[:, idx])


def test_equal_parents():
    simulator = BreedingSimulator(
        genetic_map=DATA_PATH.joinpath("small_genetic_map.txt")
    )

    n_markers = len(simulator.marker_effects)

    parents = np.zeros((1, 2, n_markers, 2), dtype="bool")
    child = simulator.cross(parents)
    assert np.all(child == 0)

    parents = np.ones((1, 2, n_markers, 2), dtype="bool")
    child = simulator.cross(parents)
    assert np.all(child == 1)
