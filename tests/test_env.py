import gymnasium as gym
import numpy as np
import pytest
from chromax.sample_data import genetic_map, genome


def test_reset_population():
    env = gym.make(
        "BreedGym",
        initial_population=genome,
        genetic_map=genetic_map,
    )

    pop, _ = env.reset()
    init_pop = np.copy(pop)

    env.step(np.asarray(env.action_space.sample()) % len(pop))
    pop, _ = env.reset()
    assert np.all(init_pop == pop)


@pytest.mark.parametrize("n", [1, 5, 10])
def test_num_progenies(n):
    env = gym.make(
        "BreedGym",
        initial_population=genome,
        genetic_map=genetic_map,
    )
    pop, _ = env.reset()

    action = np.random.randint(len(pop), size=(n, 2))
    env.step(action)

    assert len(env.population) == n


def test_caching():
    env = gym.make(
        "BreedGym",
        initial_population=genome,
        genetic_map=genetic_map,
    )
    env.reset()

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


def test_reward_shaping():
    env = gym.make(
        "BreedGym",
        initial_population=genome,
        genetic_map=genetic_map,
        reward_shaping=False,
    )

    pop, _ = env.reset()

    for _ in range(9):
        action = np.asarray(env.action_space.sample()) % len(pop)
        pop, reward, _, truncated, _ = env.step(action)

        assert reward == 0
        assert not truncated

    action = np.asarray(env.action_space.sample()) % len(pop)
    _, reward, _, truncated, _ = env.step(action)

    assert reward != 0
    assert truncated

    env2 = gym.make(
        "BreedGym",
        initial_population=genome,
        genetic_map=genetic_map,
        reward_shaping=True,
    )

    pop, _ = env2.reset()
    action = np.asarray(env2.action_space.sample()) % len(pop)
    _, reward, _, _, _ = env2.step(action)

    assert reward != 0


def test_deterministic():
    env = gym.make(
        "BreedGym",
        initial_population=genome,
        genetic_map=genetic_map,
        reward_shaping=False,
    )

    env.reset(seed=7)
    action = np.array(
        [[1, 2], [1, 5], [1, 7], [2, 5], [2, 9], [4, 7], [4, 8], [5, 9], [6, 8], [6, 9]]
    )

    for _ in range(10):
        _, r, _, _, _ = env.step(action)

    assert abs(r - (1.6602371)) < 1e-6
