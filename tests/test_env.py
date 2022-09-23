import gym
import numpy as np
from breeding_gym.utils.paths import DATA_PATH
import pytest


def test_reset_population():
    env = gym.make("BreedingGym",
                   initial_population=DATA_PATH.joinpath("small_geno.txt"),
                   genetic_map=DATA_PATH.joinpath("small_genetic_map.txt"),
                   )

    pop = env.reset(return_info=False)
    init_pop = np.copy(pop)

    env.step(np.asarray(env.action_space.sample()) % len(pop))
    pop = env.reset(return_info=False)
    assert np.all(init_pop == pop)


@pytest.mark.parametrize("n", [1, 5, 10])
def test_num_progenies(n):
    env = gym.make("BreedingGym",
                   initial_population=DATA_PATH.joinpath("small_geno.txt"),
                   genetic_map=DATA_PATH.joinpath("small_genetic_map.txt"),
                   )
    pop = env.reset(return_info=False)

    action = np.random.randint(len(pop), size=(n, 2))
    env.step(action)

    assert len(env.population) == n


def test_caching():
    env = gym.make("BreedingGym",
                   initial_population=DATA_PATH.joinpath("small_geno.txt"),
                   genetic_map=DATA_PATH.joinpath("small_genetic_map.txt"),
                   )
    env.reset(return_info=False)

    GEBV = env.GEBV
    GEBV_copy = np.copy(GEBV)
    GEBV2 = env.GEBV
    assert id(GEBV) == id(GEBV2)
    assert np.all(GEBV_copy == GEBV2)

    corrcoef = env.corrcoef
    corrcoef_copy = np.copy(corrcoef)
    corrcoef2 = env.corrcoef
    assert id(corrcoef) == id(corrcoef2)
    assert np.all(corrcoef_copy == corrcoef2)

    action = np.array([[1, 3], [4, 2]])
    env.step(action)

    GEBV3 = env.GEBV
    corrcoef3 = env.corrcoef
    assert id(corrcoef) != id(corrcoef3)
    assert id(GEBV) != id(GEBV3)
