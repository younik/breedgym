import gym
import numpy as np
from breeding_gym.utils.plot_utils import set_up_plt, NEURIPS_FONT_FAMILY


if __name__ == '__main__':
    num_generations = 10
    bests = [10, 67, 23]
    episode_names = [f"{b} bests" for b in bests]

    env = gym.make("SimplifiedBreedingGym",
                   render_mode="matplotlib",
                   render_kwargs={"episode_names": episode_names},
                   new_step_api=True
                   )

    set_up_plt(NEURIPS_FONT_FAMILY)

    for action in bests:
        pop, info = env.reset(return_info=True)

        for i in np.arange(num_generations):
            pop, r, terminated, truncated, info = env.step(action)

            mean_GEBV = np.mean(info["GEBV"], axis=0)
            print("------- GEBV -------")
            print(mean_GEBV)

    env.render()
