import pytest
from breeding_gym.utils.paths import DATA_PATH
from breeding_gym.vector.vec_env import DistributedBreedingGym, VecBreedingGym
import numpy as np
import jax
from breeding_gym.vector.vec_wrappers import PairScores, SelectionValues
import warnings
from chromax.sample_data import genome, genetic_map


def test_vec():
    n_envs = 8
    individual_per_gen = 200
    env = VecBreedingGym(
        n_envs=n_envs,
        initial_population=genome,
        genetic_map=genetic_map,
        individual_per_gen=individual_per_gen
    )

    pop, _ = env.reset()
    expected_shape = (n_envs, individual_per_gen, env.simulator.n_markers, 2)
    assert pop.shape == expected_shape

    actions = np.random.randint(
        0, individual_per_gen,
        size=(n_envs, individual_per_gen, 2)
    )
    new_pop, reward, terminated, truncated, infos = env.step(actions)

    assert new_pop.shape == expected_shape
    assert reward.shape == (n_envs,)
    assert np.all(~terminated)
    assert np.all(~truncated)
    assert isinstance(infos, dict)
    assert len(infos["GEBV"]) == n_envs
    for info in infos["GEBV"]:
        assert info.shape == (individual_per_gen, 7)


def test_selection_vec():
    n_envs = 8
    individual_per_gen = 210
    env = VecBreedingGym(
        n_envs=n_envs,
        initial_population=genome,
        genetic_map=genetic_map,
        individual_per_gen=individual_per_gen,
        trait_names=["Yield"]
    )
    env = SelectionValues(env, k=10)

    pop, _ = env.reset()
    expected_shape = (n_envs, individual_per_gen, env.simulator.n_markers, 2)
    assert pop.shape == expected_shape

    actions = np.random.rand(n_envs, individual_per_gen)
    new_pop, reward, terminated, truncated, infos = env.step(actions)
    assert new_pop.shape == expected_shape
    assert reward.shape == (n_envs,)
    assert np.all(~terminated)
    assert np.all(~truncated)
    assert isinstance(infos, dict)
    assert len(infos["GEBV"]) == n_envs
    for info in infos["GEBV"]:
        assert info.shape == (individual_per_gen, 1)


def test_distributed_env():
    local_devices = jax.local_devices()
    if len(local_devices) == 1:
        warnings.warn(
            "Distributed test skipped because there is only one device."
        )
        return

    envs_per_device = 4
    individual_per_gen = 200
    env = DistributedBreedingGym(
        envs_per_device=envs_per_device,
        devices=local_devices,
        initial_population=genome,
        genetic_map=genetic_map,
        individual_per_gen=individual_per_gen
    )
    n_envs = envs_per_device * len(local_devices)
    assert env.num_envs == n_envs
    assert env.observation_space.shape[0] == n_envs
    assert len(env.action_space) == n_envs

    pop, _ = env.reset()
    expected_shape = (n_envs, individual_per_gen, 10_000, 2)
    assert pop.shape == expected_shape

    actions = np.random.randint(
        0, individual_per_gen,
        size=(n_envs, individual_per_gen, 2)
    )
    new_pop, reward, terminated, truncated, infos = env.step(actions)

    assert new_pop.shape == expected_shape
    assert reward.shape == (n_envs,)
    assert terminated.shape == (n_envs, )
    assert truncated.shape == (n_envs, )
    assert np.all(~terminated)
    assert np.all(~truncated)
    assert isinstance(infos, dict)
    assert len(infos["GEBV"]) == n_envs
    for info in infos["GEBV"]:
        assert info.shape == (individual_per_gen, 1)

    env.close()


def test_vec_deterministic():
    n_envs = 4
    individual_per_gen = 200
    env = VecBreedingGym(
        n_envs=n_envs,
        initial_population=genome,
        genetic_map=genetic_map,
        individual_per_gen=individual_per_gen
    )

    np.random.seed(seed=7)
    pop, _ = env.reset(seed=7)
    for _ in range(20):
        action = np.random.randint(
            len(pop), size=(n_envs, individual_per_gen, 2)
        )
        pop, rews, _, _, _ = env.step(action)

    expected_result = np.array([11.627516, 10.363208, 12.290365, 10.221108])
    assert np.allclose(rews, expected_result)


def test_vec_gebv_policy():
    n_envs = 4
    individual_per_gen = 200
    env = VecBreedingGym(
        n_envs=n_envs,
        initial_population=genome,
        genetic_map=genetic_map,
        individual_per_gen=individual_per_gen,
        trait_names=["Yield"]
    )
    env = SelectionValues(env, k=10)

    _, infos = env.reset(seed=7)
    for _ in range(10):
        _, rews, _, _, infos = env.step(infos["GEBV"].squeeze())

    expected_result = np.array([20.377941, 21.639626, 20.222626, 21.639626])
    assert np.allclose(rews, expected_result)


def test_vec_wrapper_n_crosses():
    n_envs = 4
    individual_per_gen = 200
    env = VecBreedingGym(
        n_envs=n_envs,
        initial_population=genome,
        genetic_map=genetic_map,
        individual_per_gen=individual_per_gen,
        trait_names=["Yield"]
    )

    env = SelectionValues(env, k=10, n_crosses=20)
    _, infos = env.reset(seed=7)
    pop, _, _, _, _ = env.step(infos["GEBV"].squeeze())
    assert pop.shape[1] == individual_per_gen

    with pytest.raises(ValueError):
        env = SelectionValues(env, k=100, n_crosses=201)

    with pytest.raises(ValueError):
        env = SelectionValues(env, k=2, n_crosses=10)

    with pytest.raises(ValueError):
        env = SelectionValues(env, k=1, n_crosses=1)

    with pytest.raises(ValueError):
        env = SelectionValues(env, k=500, n_crosses=10)


def test_vec_pair_score():
    n_envs = 4
    individual_per_gen = 200
    env = VecBreedingGym(
        n_envs=n_envs,
        initial_population=genome,
        genetic_map=genetic_map,
        individual_per_gen=individual_per_gen
    )
    env = PairScores(env)

    _, infos = env.reset(seed=7)
    for _ in range(10):
        gebvs = infos["GEBV"].squeeze()
        gebvs_matrix_sum = np.add.outer(gebvs, gebvs)
        _, rews, _, _, infos = env.step(gebvs_matrix_sum)

    expected_result = np.array([13.073561,  7.534841, 11.851914, 15.998689])
    assert np.allclose(rews, expected_result)
