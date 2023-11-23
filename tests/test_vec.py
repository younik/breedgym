import warnings

import gymnasium as gym
import jax
import numpy as np
import pytest
from chromax.sample_data import genetic_map, genome

from breedgym.vector.vec_env import DistributedBreedGym, VecBreedGym
from breedgym.vector.vec_wrappers import SelectionScores


def test_vec():
    num_envs = 8
    individual_per_gen = 200
    env = gym.make(
        "VecBreedGym",
        num_envs=num_envs,
        initial_population=genome,
        genetic_map=genetic_map,
        individual_per_gen=individual_per_gen,
    )

    pop, _ = env.reset()
    expected_shape = (num_envs, individual_per_gen, env.simulator.n_markers, 2)
    assert pop.shape == expected_shape

    actions = np.random.randint(
        0, individual_per_gen, size=(num_envs, individual_per_gen, 2)
    )
    new_pop, reward, terminated, truncated, infos = env.step(actions)

    assert new_pop.shape == expected_shape
    assert reward.shape == (num_envs,)
    assert np.all(~terminated)
    assert np.all(~truncated)
    assert isinstance(infos, dict)
    assert len(infos["GEBV"]) == num_envs
    for info in infos["GEBV"]:
        assert info.shape == (individual_per_gen, 7)


def test_selection_vec():
    num_envs = 8
    individual_per_gen = 210
    env = gym.make(
        "SelectionScores",
        k=10,
        num_envs=num_envs,
        initial_population=genome,
        genetic_map=genetic_map,
        individual_per_gen=individual_per_gen,
        trait_names=["Yield"],
    )

    pop, _ = env.reset()
    expected_shape = (num_envs, individual_per_gen, env.simulator.n_markers, 2)
    assert pop.shape == expected_shape

    actions = np.random.rand(num_envs, individual_per_gen)
    new_pop, reward, terminated, truncated, infos = env.step(actions)
    assert new_pop.shape == expected_shape
    assert reward.shape == (num_envs,)
    assert np.all(~terminated)
    assert np.all(~truncated)
    assert isinstance(infos, dict)
    assert len(infos["GEBV"]) == num_envs
    for info in infos["GEBV"]:
        assert info.shape == (individual_per_gen, 1)


def test_distributed_env():
    local_devices = jax.local_devices()
    if len(local_devices) == 1:
        warnings.warn("Distributed test skipped because there is only one device.")
        return

    envs_per_device = 4
    individual_per_gen = 200
    env = DistributedBreedGym(
        envs_per_device=envs_per_device,
        devices=local_devices,
        initial_population=genome,
        genetic_map=genetic_map,
        individual_per_gen=individual_per_gen,
    )
    num_envs = envs_per_device * len(local_devices)
    assert env.num_envs == num_envs
    assert env.observation_space.shape[0] == num_envs
    assert len(env.action_space) == num_envs

    pop, _ = env.reset()
    expected_shape = (num_envs, individual_per_gen, 10_000, 2)
    assert pop.shape == expected_shape

    actions = np.random.randint(
        0, individual_per_gen, size=(num_envs, individual_per_gen, 2)
    )
    new_pop, reward, terminated, truncated, infos = env.step(actions)

    assert new_pop.shape == expected_shape
    assert reward.shape == (num_envs,)
    assert terminated.shape == (num_envs,)
    assert truncated.shape == (num_envs,)
    assert np.all(~terminated)
    assert np.all(~truncated)
    assert isinstance(infos, dict)
    assert len(infos["GEBV"]) == num_envs
    for info in infos["GEBV"]:
        assert info.shape == (individual_per_gen, 1)

    env.close()


def test_vec_deterministic():
    num_envs = 4
    individual_per_gen = 200
    env = gym.make(
        "VecBreedGym",
        num_envs=num_envs,
        initial_population=genome,
        genetic_map=genetic_map,
        individual_per_gen=individual_per_gen,
    )

    np.random.seed(seed=7)
    pop, _ = env.reset(seed=7)
    for _ in range(20):
        action = np.random.randint(len(pop), size=(num_envs, individual_per_gen, 2))
        pop, rews, _, _, _ = env.step(action)

    expected_result = np.array([11.618341, 11.29245, 13.47926, 10.03809])
    assert np.allclose(rews, expected_result)


def test_vec_gebv_policy():
    num_envs = 4
    individual_per_gen = 200
    env = gym.make(
        "SelectionScores",
        k=10,
        num_envs=num_envs,
        initial_population=genome,
        genetic_map=genetic_map,
        individual_per_gen=individual_per_gen,
        trait_names=["Yield"],
    )

    _, infos = env.reset(seed=7)
    for _ in range(10):
        _, rews, _, _, infos = env.step(infos["GEBV"].squeeze())

    expected_result = np.array([20.43854, 21.59488, 20.503202, 21.617622])
    assert np.allclose(rews, expected_result)


def test_vec_wrapper_n_crosses():
    num_envs = 4
    individual_per_gen = 200
    env = VecBreedGym(
        num_envs=num_envs,
        initial_population=genome,
        genetic_map=genetic_map,
        individual_per_gen=individual_per_gen,
        trait_names=["Yield"],
    )

    wrap_env = SelectionScores(env, k=10, n_crosses=20)
    _, infos = wrap_env.reset(seed=7)
    pop, _, _, _, _ = wrap_env.step(infos["GEBV"].squeeze())
    assert pop.shape[1] == individual_per_gen

    with pytest.raises(ValueError):
        wrap_env = SelectionScores(env, k=100, n_crosses=201)

    with pytest.raises(ValueError):
        wrap_env = SelectionScores(env, k=2, n_crosses=10)

    with pytest.raises(ValueError):
        wrap_env = SelectionScores(env, k=1, n_crosses=1)

    with pytest.raises(ValueError):
        wrap_env = SelectionScores(env, k=500, n_crosses=10)


def test_vec_pair_score():
    num_envs = 4
    individual_per_gen = 200
    env = gym.make(
        "PairScores",
        num_envs=num_envs,
        initial_population=genome,
        genetic_map=genetic_map,
        individual_per_gen=individual_per_gen,
    )

    _, infos = env.reset(seed=7)
    for _ in range(10):
        gebvs = infos["GEBV"].squeeze()
        gebvs_matrix_sum = np.add.outer(gebvs, gebvs)
        _, rews, _, _, infos = env.step(gebvs_matrix_sum)

    expected_result = np.array([13.162314, 10.155096, 12.45827, 12.093216])
    assert np.allclose(rews, expected_result)
