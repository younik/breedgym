from breeding_gym.baseline_agent import BaselineAgent, default_f_index
import gym
import numpy as np

dir_path = "/home/omar/PycharmProjects/breedinggym/breeding_gym"
env = gym.make("BreedingGym",
    chromosomes_map=f"{dir_path}/data/map.txt", 
    population=f"{dir_path}/data/geno.txt", 
    marker_effects=f"{dir_path}/data/marker_effects.txt",

    new_step_api = True
)
agent = BaselineAgent()

pop, info = env.reset(return_info=True)

for _ in range(10):
    pop, r, terminated, truncated, info = env.step(agent(info["GEBV"]))
    mean_GEBV = np.mean(info["GEBV"], axis=0)
    print("GEBV:", mean_GEBV)
    print("Index:", default_f_index(mean_GEBV[None, :]))
    #print("corrcoef:", np.mean(info["corrcoef"][info["corrcoef"] != 1]))