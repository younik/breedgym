import gym
from breeding_gym.utils.paths import DATA_PATH
import pytest
import numpy as np


def test_simplified_env():
    individual_per_gen = 200
    env = gym.make("SimplifiedBreedingGym",
                   individual_per_gen=individual_per_gen,
                   initial_population=DATA_PATH.joinpath("small_geno.txt"),
                   genetic_map=DATA_PATH.joinpath("small_genetic_map.txt"),
                   )

    obs, _ = env.reset()
    assert len(obs["GEBV"]) == individual_per_gen
    assert len(obs["corrcoef"]) == individual_per_gen
    assert np.all(obs["corrcoef"] >= -1) and np.all(obs["corrcoef"] <= 1)

    actions = [
        {"n_bests": 10, "n_crosses": 20},
        {"n_bests": 21, "n_crosses": 200},
        {"n_bests": 2, "n_crosses": 1}
    ]

    for action in actions:
        obs, _, _, _, _ = env.step(action)
        assert len(obs["GEBV"]) == individual_per_gen
        assert len(obs["corrcoef"]) == individual_per_gen
        assert np.all(obs["corrcoef"] >= -1) and np.all(obs["corrcoef"] <= 1)

    with pytest.raises(Exception):
        env.step({"n_bests": 100, "n_crosses": 201})

    with pytest.raises(Exception):
        env.step({"n_bests": 2, "n_crosses": 10})

    with pytest.raises(Exception):
        env.step({"n_bests": 1, "n_crosses": 1})

    with pytest.raises(Exception):
        env.step({"n_bests": 500, "n_crosses": 10})


def test_kbest_env():
    individual_per_gen = 200
    env = gym.make("KBestBreedingGym",
                   individual_per_gen=individual_per_gen,
                   initial_population=DATA_PATH.joinpath("small_geno.txt"),
                   genetic_map=DATA_PATH.joinpath("small_genetic_map.txt"),
                   )
    obs, _ = env.reset()
    assert len(obs["GEBV"]) == individual_per_gen
    assert len(obs["corrcoef"]) == individual_per_gen
    assert np.all(obs["corrcoef"] >= -1) and np.all(obs["corrcoef"] <= 1)

    actions = [10, 2, 20]
    for action in actions:
        obs, _, _, _, _ = env.step(action)
        assert len(obs["GEBV"]) == individual_per_gen
        assert len(obs["corrcoef"]) == individual_per_gen
        assert np.all(obs["corrcoef"] >= -1) and np.all(obs["corrcoef"] <= 1)

    with pytest.raises(Exception):
        env.step(1)

    with pytest.raises(Exception):
        env.step(21)
