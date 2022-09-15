import gym
import breeding_gym  # noqa
import numpy as np
from breeding_gym.utils.index_functions import optimal_haploid_value, yield_index
from breeding_gym.utils.paths import DATA_PATH

if __name__ == '__main__':
    num_generations = 10
    bests = [10]
    episode_names = [f"{b} bests" for b in bests]

    env = gym.make("SimplifiedBreedingGym",
                   individual_per_gen=200,
                   initial_population=DATA_PATH.joinpath("small_geno.txt"),
                   genetic_map=DATA_PATH.joinpath("small_genetic_map.txt"),
                   render_mode="matplotlib",
                   render_kwargs={"episode_names": ["yield", "OHV"]},
                   #render_kwargs={"episode_names": episode_names},
                   new_step_api=True
                   )

    for action in bests:
        pop, info = env.reset(return_info=True, options={"index": yield_index})

        for i in np.arange(num_generations):
            pop, r, terminated, truncated, info = env.step(action)

            mean_GEBV = np.mean(info["GEBV"], axis=0)
            print("------- GEBV -------")
            print(mean_GEBV)

        pop, info = env.reset(return_info=True, options={"index": optimal_haploid_value})
        for i in np.arange(num_generations):
            pop, r, terminated, truncated, info = env.step(action)

            mean_GEBV = np.mean(info["GEBV"], axis=0)
            print("------- GEBV -------")
            print(mean_GEBV)

    env.render()
