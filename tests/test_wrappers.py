import gym
from breeding_gym.utils.paths import DATA_PATH
import pytest

from breeding_gym.wrappers import ObserveStepWrapper


def test_simplified_env():
    env = gym.make("SimplifiedBreedingGym",
                   individual_per_gen=200,
                   initial_population=DATA_PATH.joinpath("small_geno.txt"),
                   genetic_map=DATA_PATH.joinpath("small_genetic_map.txt"),
                   )
    env.reset()
    env.step({"n_bests": 10, "n_crosses": 20})
    env.step({"n_bests": 21, "n_crosses": 200})
    env.step({"n_bests": 2, "n_crosses": 1})

    with pytest.raises(Exception):
        env.step({"n_bests": 2, "n_crosses": 10})

    with pytest.raises(Exception):
        env.step({"n_bests": 1, "n_crosses": 1})

    with pytest.raises(Exception):
        env.step({"n_bests": 500, "n_crosses": 10})


def test_kbest_env():
    env = gym.make("KBestBreedingGym",
                   individual_per_gen=200,
                   initial_population=DATA_PATH.joinpath("small_geno.txt"),
                   genetic_map=DATA_PATH.joinpath("small_genetic_map.txt"),
                   )
    env.reset()
    env.step(10)
    env.step(2)
    env.step(200)

    with pytest.raises(Exception):
        env.step(1)

    with pytest.raises(Exception):
        env.step(201)


def test_observe_wrapper():
    env = gym.make("SimplifiedBreedingGym",
                   individual_per_gen=200,
                   initial_population=DATA_PATH.joinpath("small_geno.txt"),
                   genetic_map=DATA_PATH.joinpath("small_genetic_map.txt"),
                   )
    env = ObserveStepWrapper(env)

    obs, _ = env.reset()
    assert obs == 0

    obs, _, _, _, _ = env.step({"n_bests": 10, "n_crosses": 20})
    assert obs == 1

    obs, _ = env.reset()
    assert obs == 0
