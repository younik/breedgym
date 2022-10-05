import gym
import breeding_gym  # noqa
import numpy as np
from breeding_gym.utils.index_functions import (
    optimal_haploid_value,
    yield_index
)
from breeding_gym.utils.paths import DATA_PATH

if __name__ == '__main__':
    num_generations = 10
    bests = [
        {"n_bests": 20, "n_crosses": 10},
    ]
    episode_names = [f"{b} bests" for b in bests]

    env = gym.make("SimplifiedBreedingGym",
                   individual_per_gen=200,
                   initial_population=DATA_PATH.joinpath("medium_geno.txt"),
                   genetic_map=DATA_PATH.joinpath("medium_genetic_map.txt"),
                   render_mode="matplotlib",
                   render_kwargs={
                       "episode_names": ["yield", "OHV"],
                    },
                   )

    for action in bests:
        pop, info = env.reset(options={"index": yield_index})

        for i in np.arange(num_generations):
            pop, r, terminated, truncated, info = env.step(action)

            mean_GEBV = np.mean(info["GEBV"], axis=0)
            print("------- GEBV -------", flush=True)
            print(mean_GEBV, flush=True)

        pop, info = env.reset(options={"index": optimal_haploid_value})
        for i in np.arange(num_generations):
            pop, r, terminated, truncated, info = env.step(action)

            mean_GEBV = np.mean(info["GEBV"], axis=0)
            print("------- GEBV -------", flush=True)
            print(mean_GEBV, flush=True)

    env.render(file_name="figures/fixed_policy_example.png")
