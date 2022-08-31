from breeding_gym.baseline_agent import BaselineAgent
import gym
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    env = gym.make("BreedingGym", new_step_api=True)

    num_generations = 10
    offspring_list = [10, 20, 50]
    colors = ['b', 'g', 'r']

    boxplot_elements = [
        'boxes',
        'whiskers',
        'fliers',
        'means',
        'medians',
        'caps'
    ]

    fig, axs = plt.subplots(1, 2)

    for offset, (n_offspring, c) in enumerate(zip(offspring_list, colors)):
        agent = BaselineAgent(n_offspring=n_offspring)
        pop, info = env.reset(return_info=True)
        for i in np.arange(num_generations):
            pop, r, terminated, truncated, info = env.step(agent(info["GEBV"]))

            yield_ = info["GEBV"][:, 0]
            bp = axs[0].boxplot(yield_, positions=[i + 1 + (offset - 1) / 5])
            for element in boxplot_elements:
                plt.setp(bp[element], color=c)

            mean_GEBV = np.mean(info["GEBV"], axis=0)
            print("GEBV:", mean_GEBV)

            corrcoef = env.corrcoef()[:, 0]
            bp = axs[1].boxplot(corrcoef, positions=[i + 1 + (offset - 1) / 5])
            for element in boxplot_elements:
                plt.setp(bp[element], color=c)

    xticks = np.arange(num_generations) + 1
    axs[0].set_xticks(xticks, xticks)
    axs[1].set_xticks(xticks, xticks)
    plt.show()
