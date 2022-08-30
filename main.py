from breeding_gym.baseline_agent import BaselineAgent
from breeding_gym.paths import DATA_PATH
import gym
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    env = gym.make("BreedingGym",
                   chromosomes_map=DATA_PATH.joinpath("map.txt"),
                   population=DATA_PATH.joinpath("geno.txt"),
                   marker_effects=DATA_PATH.joinpath("marker_effects.txt"),
                   new_step_api=True)

    num_generations = 10
    fig, axs = plt.subplots(1, 2)

    for offset, (n_offspring, c) in enumerate(zip([10, 20, 50], ['b', 'g', 'r'])):
        agent = BaselineAgent(n_offspring=n_offspring)
        pop, info = env.reset(return_info=True)
        for i in np.arange(num_generations):
            pop, r, terminated, truncated, info = env.step(agent(info["GEBV"]))
            bp = axs[0].boxplot(info["GEBV"][:, 0], positions=[i+1 + (offset-1)/5])
            for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
                plt.setp(bp[element], color=c)
            mean_GEBV = np.mean(info["GEBV"], axis=0)
            print("GEBV:", mean_GEBV)
            bp = axs[1].boxplot(info["corrcoef"][:, 0], positions=[i + 1 + (offset - 1) / 5])
            for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
                plt.setp(bp[element], color=c)

    axs[0].set_xticks(np.arange(num_generations)+1, np.arange(num_generations)+1)
    axs[1].set_xticks(np.arange(num_generations)+1, np.arange(num_generations)+1)
    plt.show()