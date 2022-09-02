import gym
import breeding_gym  # noqa: F401
import numpy as np

env = gym.make("SimplifiedBreedingGym", new_step_api=True)

pop, info = env.reset(return_info=True)

for _ in range(10):
    pop, r, terminated, truncated, info = env.step(10)
    mean_GEBV = np.mean(info["GEBV"], axis=0)
    print("GEBV:", mean_GEBV)
    # print("Index:", default_f_index(mean_GEBV[None, :]))
