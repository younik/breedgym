from breeding_gym.baseline_agent import BaselineAgent, default_f_index
import gym
import numpy as np

env = gym.make("BreedingGym", new_step_api=True)

# decision_index = partial(optimal_haploid_value, env)
agent = BaselineAgent()

pop, info = env.reset(return_info=True)

for _ in range(10):
    pop, r, terminated, truncated, info = env.step(agent(info["GEBV"]))
    mean_GEBV = np.mean(info["GEBV"], axis=0)
    print("GEBV:", mean_GEBV)
    print("Index:", default_f_index(mean_GEBV[None, :]))
