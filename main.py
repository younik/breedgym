from breeding_gym.baseline_agent import BaselineAgent
from breeding_gym.paths import DATA_PATH, FIGURE_PATH
from breeding_gym.plot_utils import set_up_plt, NEURIPS_FONT_FAMILY
import gym
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


if __name__ == '__main__':
    env = gym.make("BreedingGym",
                   chromosomes_map=DATA_PATH.joinpath("map.txt"),
                   population=DATA_PATH.joinpath("geno.txt"),
                   marker_effects=DATA_PATH.joinpath("marker_effects.txt"),
                   new_step_api=True)

    num_generations = 10
    set_up_plt(NEURIPS_FONT_FAMILY)
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    n_offsprings = [10, 50, 100]
    colors = ['b', 'g', 'r']
    for offset, (n_offspring, c) in enumerate(zip(n_offsprings, colors)):
        agent = BaselineAgent(n_offspring=n_offspring)
        pop, info = env.reset(return_info=True)
        for i in np.arange(num_generations):
            pop, r, terminated, truncated, info = env.step(agent(info["GEBV"]))
            bp = axs[0].boxplot(info["GEBV"][:, 0], positions=[i+1 + (offset-1)/5], flierprops={'markersize': 2})
            for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
                plt.setp(bp[element], color=c)
            mean_GEBV = np.mean(info["GEBV"], axis=0)
            print("GEBV:", mean_GEBV)
            bp = axs[1].boxplot(1 - info["corrcoef"][:, 0], positions=[i + 1 + (offset - 1) / 5],
                                flierprops={'markersize': 2})
            for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
                plt.setp(bp[element], color=c)

    axs[0].set_xticks(np.arange(num_generations)+1, np.arange(num_generations)+1)
    axs[1].set_xticks(np.arange(num_generations)+1, np.arange(num_generations)+1)
    axs[0].set_title('Crop Yield [Ton/ha]')
    axs[1].set_title('Genetic variability of population')
    axs[0].grid(axis='y')
    axs[1].grid(axis='y')
    axs[0].set_xlabel('Generations [Years]')
    axs[0].xaxis.set_label_coords(1.1, -0.07)
    patches = [mpatches.Rectangle((0, 1), color=c, label=o, width=0.1, height=0.1, fill=False)
               for o, c in zip(n_offsprings, colors)]
    axs[1].legend(handles=patches, loc='upper right')
    plt.savefig(FIGURE_PATH.joinpath('boxplots.png'), bbox_inches='tight')
    plt.show()