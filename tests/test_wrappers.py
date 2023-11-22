import gymnasium as gym
import numpy as np
import pytest
from chromax.sample_data import genetic_map, genome


def test_simplified_env():
    individual_per_gen = 200
    env = gym.make(
        "SimplifiedBreedGym",
        individual_per_gen=individual_per_gen,
        initial_population=genome,
        genetic_map=genetic_map,
    )

    obs, _ = env.reset()
    assert len(obs["GEBV"]) == individual_per_gen
    assert len(obs["corrcoef"]) == individual_per_gen
    assert np.all(obs["corrcoef"] >= -1) and np.all(obs["corrcoef"] <= 1)

    actions = [
        {"n_bests": 10, "n_crosses": 20},
        {"n_bests": 21, "n_crosses": 200},
        {"n_bests": 2, "n_crosses": 1},
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
    env = gym.make(
        "KBestBreedGym",
        individual_per_gen=individual_per_gen,
        initial_population=genome,
        genetic_map=genetic_map,
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


def test_gebv_policy():
    individual_per_gen = 200
    env = gym.make(
        "KBestBreedGym",
        individual_per_gen=individual_per_gen,
        initial_population=genome,
        genetic_map=genetic_map,
        trait_names=["Yield"],
    )
    env.reset(seed=7)
    action = 10

    for _ in range(10):
        _, r, _, _, _ = env.step(action)

    assert abs(r - 20.315035) < 1e-5
