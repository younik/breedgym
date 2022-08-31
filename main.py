from breeding_gym.baseline_agent import BaselineAgent
from breeding_gym.paths import FIGURE_PATH
from breeding_gym.plot_utils import set_up_plt, NEURIPS_FONT_FAMILY
import gym
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


if __name__ == '__main__':
    env = gym.make("BreedingGym", new_step_api=True)

    num_generations = 10
    n_offsprings = [10, 50, 100]

    set_up_plt(NEURIPS_FONT_FAMILY)
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    colors = ['b', 'g', 'r']
    boxplot_elements = [
        'boxes',
        'whiskers',
        'fliers',
        'means',
        'medians',
        'caps'
    ]

    for offset, (n_offspring, c) in enumerate(zip(n_offsprings, colors)):
        agent = BaselineAgent(n_offspring=n_offspring)
        pop, info = env.reset(return_info=True)
        for i in np.arange(num_generations):
            action = agent(info["GEBV"])
            pop, r, terminated, truncated, info = env.step(action)

            positions = [i + 1 + (offset - 1) / 5]
            yield_ = info["GEBV"][:, 0]
            bp = axs[0].boxplot(
                yield_,
                positions=positions,
                flierprops={'markersize': 2}
            )
            for element in boxplot_elements:
                plt.setp(bp[element], color=c)

            mean_GEBV = np.mean(info["GEBV"], axis=0)
            print("GEBV:", mean_GEBV)

            corrcoeff = env.corrcoef()[:, 0]
            bp = axs[1].boxplot(
                1 - corrcoeff,
                positions=positions,
                flierprops={'markersize': 2}
            )
            for element in boxplot_elements:
                plt.setp(bp[element], color=c)

    xticks = np.arange(num_generations) + 1
    axs[0].set_xticks(xticks, xticks)
    axs[1].set_xticks(xticks, xticks)
    axs[0].set_title('Crop Yield [Ton/ha]')
    axs[1].set_title('Genetic variability of population')
    axs[0].grid(axis='y')
    axs[1].grid(axis='y')
    axs[0].set_xlabel('Generations [Years]')
    axs[0].xaxis.set_label_coords(1.1, -0.07)

    rectangle_kwargs = {"width": 0.1, "height": 0.1, "fill": False}
    patches = [mpatches.Rectangle((0, 1), color=c, label=o, **rectangle_kwargs)
               for o, c in zip(n_offsprings, colors)]
    axs[1].legend(handles=patches, loc='upper right')

    plt.savefig(FIGURE_PATH.joinpath('boxplots.png'), bbox_inches='tight')
    plt.show()
