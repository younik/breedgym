from breeding_gym.simulator import BreedingSimulator
import pytest
import numpy as np
from breeding_gym.utils.paths import DATA_PATH
import gym


@pytest.mark.parametrize("idx", [0, 1])
def test_cross_r(idx):    
    env = gym.make("BreedingGym",
                initial_population=DATA_PATH.joinpath("small_geno.txt"),
                genetic_map=DATA_PATH.joinpath("small_genetic_map.txt"),
                )
    
    def const_co_mask(n_progenies):
        shape = (n_progenies, 2, env.simulator.n_markers)
        return idx * np.ones(shape, dtype="bool")
    
    env.simulator._get_crossover_mask = const_co_mask
    init_pop = env.reset()
    p0, p1 = init_pop[0], init_pop[1]
    assert p0.shape == p1.shape

    action = np.array([[0, 1]])
    new_pop, _, _, _ = env.step(action)

    assert new_pop.shape == (1, p0.shape[0], 2)

    ind = new_pop[0]
    assert np.all(ind[:, 0] == p0[:, idx])
    assert np.all(ind[:, 1] == p1[:, idx])
        

def test_equal_parents():
    simulator = BreedingSimulator(
        genetic_map=DATA_PATH.joinpath("small_genetic_map.txt")
    )

    parents = np.zeros((1, 2, simulator.n_markers, 2), dtype="bool")
    child = simulator.cross(parents)
    assert np.all(child == 0)

    parents = np.ones((1, 2, simulator.n_markers, 2), dtype="bool")
    child = simulator.cross(parents)
    print(child)
    assert np.all(child == 1)
