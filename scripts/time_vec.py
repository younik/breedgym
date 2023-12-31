import timeit
from functools import partial

import jax
import numpy as np
import pandas as pd
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv

from breedgym.breedgym import BreedGym
from breedgym.utils.paths import DATA_PATH
from breedgym.vector.vec_env import DistributedBreedGym, VecBreedGym


def make_jax_vec(n_env):
    env = VecBreedGym(
        num_envs=n_env,
        initial_population=DATA_PATH.joinpath("small_geno.txt"),
        genetic_map=DATA_PATH.joinpath("small_genetic_map.txt"),
        individual_per_gen=210,
    )
    return env


def make_distributed_vec(n_env):
    env = DistributedBreedGym(
        envs_per_device=n_env // jax.local_device_count(),
        initial_population=DATA_PATH.joinpath("small_geno.txt"),
        genetic_map=DATA_PATH.joinpath("small_genetic_map.txt"),
        individual_per_gen=210,
    )
    return env


def make_async_vec(n_env):
    def make_env():
        env = BreedGym(
            initial_population=DATA_PATH.joinpath("small_geno.txt"),
            genetic_map=DATA_PATH.joinpath("small_genetic_map.txt"),
        )
        env.reset = partial(env.reset, options={"n_individuals": 210})
        return env

    envs = SubprocVecEnv([make_env for _ in range(n_env)])
    return envs


if __name__ == "__main__":
    num_envs = [8, 16, 32]
    print(jax.local_devices())
    env_makers = [make_jax_vec, make_distributed_vec]
    names = ["JaxVec", "DistributedEnv"]
    repeats = 10
    table = np.empty((len(num_envs), len(env_makers)))
    for row, n_env in enumerate(num_envs):
        for col, env_maker in enumerate(env_makers):
            print("@" * 10, col, "@" * 10)
            env = env_maker(n_env)
            env.reset()

            t = timeit.timeit(
                lambda: env.step(np.random.randint(0, 210, size=(n_env, 210, 2))),
                number=repeats,
            )

            table[row, col] = t / repeats

            env.close()
            del env

    df = pd.DataFrame(table, index=num_envs, columns=names)
    print(df, flush=True)
