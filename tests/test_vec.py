from breeding_gym.utils.paths import DATA_PATH
from breeding_gym.vector.vec_env import DistributedBreedingGym, VecBreedingGym
import numpy as np
import jax
from breeding_gym.vector.vec_wrappers import SelectionValues
import warnings


def test_vec():
    n_envs = 8
    individual_per_gen = 200
    env = VecBreedingGym(
        n_envs=n_envs,
        initial_population=DATA_PATH.joinpath("small_geno.txt"),
        genetic_map=DATA_PATH.joinpath("small_genetic_map.txt"),
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
        assert info.shape == (individual_per_gen, 1)


def test_selection_vec():
    n_envs = 8
    individual_per_gen = 210
    env = VecBreedingGym(
        n_envs=n_envs,
        initial_population=DATA_PATH.joinpath("small_geno.txt"),
        genetic_map=DATA_PATH.joinpath("small_genetic_map.txt"),
        individual_per_gen=individual_per_gen
    )
    env = SelectionValues(env)

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
        initial_population=DATA_PATH.joinpath("small_geno.txt"),
        genetic_map=DATA_PATH.joinpath("small_genetic_map.txt"),
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
