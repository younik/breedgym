import gym
import breeding_gym  # noqa
import numpy as np
from breeding_gym.utils.index_functions import (
    optimal_haploid_value,
    optimal_population_value,
    yield_index
)
from breeding_gym.utils.paths import DATA_PATH
import matplotlib.pyplot as plt
from breeding_gym.utils.plot_utils import NEURIPS_FONT_FAMILY, set_up_plt

if __name__ == '__main__':
    num_generations = 10
    individual_per_gen = 200
    n_bests = 20
    n_crosses = 10

    trials = 20
    indices = [
        yield_index,
        optimal_haploid_value,
        optimal_population_value(n_bests)
    ]
    names = ["standard", "OHV", "OPV"]
    colors = ["b", "g", "r"]

    set_up_plt(NEURIPS_FONT_FAMILY, use_tex=False)
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))

    env = gym.make("SimplifiedBreedingGym",
                   individual_per_gen=individual_per_gen,
                   initial_population=DATA_PATH.joinpath("medium_geno.txt"),
                   genetic_map=DATA_PATH.joinpath("medium_genetic_map.txt"),
                   )

    for label, index, color in zip(names, indices, colors):
        buffer_gg = np.empty((trials, num_generations))
        buffer_corr = np.empty((trials, num_generations))

        for trial_idx in range(trials):
            pop, info = env.reset(options={"index": index})

            print("Trial:", trial_idx, flush=True)
            for i in range(num_generations):
                action = {"n_bests": n_bests, "n_crosses": n_crosses}
                pop, r, terminated, truncated, info = env.step(action)
                buffer_gg[trial_idx, i] = env.GEBV.mean()
                buffer_corr[trial_idx, i] = np.mean(env.corrcoef)

        gg = np.mean(buffer_gg, axis=0) / env.simulator.max_gebv * 100
        corr = np.mean(buffer_corr, axis=0)
        xticks = np.arange(num_generations)

        axs[0].set_xticks(xticks)
        axs[0].set_title("GEBV")
        axs[0].grid(axis='y')
        axs[0].set_xlabel('Generations [Years]')
        axs[0].plot(xticks, gg, label=label)

        axs[1].set_xticks(xticks)
        axs[1].set_title("corrcoef")
        axs[1].grid(axis='y')
        axs[1].set_xlabel('Generations [Years]')
        axs[1].plot(xticks, corr)


plt.figlegend(loc='upper right')
plt.savefig("figures/compare_indexes.png")
plt.show()
