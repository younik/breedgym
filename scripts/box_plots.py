from breeding_gym.utils.paths import FIGURE_PATH
from breeding_gym.utils.plot_utils import set_up_plt, NEURIPS_FONT_FAMILY
import gym
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


if __name__ == '__main__':
    env = gym.make("SimplifiedBreedingGym", new_step_api=True)

    num_generations = 10
    experiments = [
        {
            'best': 10,
            'color': 'b'
        },
        {
            'best': 67,
            'color': 'g'
        }
    ]

    set_up_plt(NEURIPS_FONT_FAMILY)
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))

    boxplot_elements = [
        'boxes',
        'whiskers',
        'fliers',
        'means',
        'medians',
        'caps'
    ]

    for offset, hyperparams in enumerate(experiments):
        pop, info = env.reset(return_info=True)

        for i in np.arange(num_generations):
            pop, r, terminated, truncated, info = env.step(hyperparams["best"])

            positions = [i + 1 + (offset - 1) / 5]
            yield_ = info["GEBV"][:, 0]
            bp = axs[0].boxplot(
                yield_,
                positions=positions,
                flierprops={'markersize': 2}
            )
            for element in boxplot_elements:
                plt.setp(bp[element], color=hyperparams["color"])

            mean_GEBV = np.mean(info["GEBV"], axis=0)
            print("GEBV:", mean_GEBV)

            corrcoeff = env.corrcoef()[:, 0]
            bp = axs[1].boxplot(
                1 - corrcoeff,
                positions=positions,
                flierprops={'markersize': 2}
            )
            for element in boxplot_elements:
                plt.setp(bp[element], color=hyperparams["color"])

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
    patches = [mpatches.Rectangle((0, 1),
                                  color=exp["color"],
                                  label=exp["best"],
                                  **rectangle_kwargs
                                  )
               for exp in experiments]
    axs[1].legend(handles=patches, loc='upper right')

    plt.savefig(FIGURE_PATH.joinpath('boxplots.png'), bbox_inches='tight')
    plt.show()
