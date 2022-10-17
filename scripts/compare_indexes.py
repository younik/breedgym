import gym
import numpy as np
from breeding_gym.simulator.gebv_model import GEBVModel
from breeding_gym.utils.index_functions import (
    optimal_haploid_value,
    optimal_population_value,
    yield_index
)
from breeding_gym.utils.paths import DATA_PATH
import matplotlib.pyplot as plt
from breeding_gym.utils.plot_utils import NEURIPS_FONT_FAMILY, set_up_plt
from breeding_gym.wrappers import SimplifiedBreedingGym


class GeneticDiversityWrapper(gym.Wrapper):

    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        genetic_diversity = self.genetic_diversity(self.population[action])
        obs, r, ter, tru, info = super().step(action)
        info["genetic_diversity"] = genetic_diversity
        return obs, r, ter, tru, info

    def genetic_diversity(self, parents: np.ndarray):
        parents = parents.transpose(0, 2, 1, 3)
        parents = parents.reshape(parents.shape[0], parents.shape[1], -1)

        max_GEBV = self.simulator.GEBV_model.optimal_haploid_value(parents)
        neg_GEBV = GEBVModel(-self.simulator.GEBV_model.marker_effects)
        min_GEBV = -neg_GEBV.optimal_haploid_value(parents)
        return max_GEBV - min_GEBV


if __name__ == '__main__':
    num_generations = 10
    individual_per_gen = 200
    n_bests = 20
    n_crosses = 10

    trials = 100
    indices = [
        yield_index,
        optimal_haploid_value,
        optimal_population_value(n_bests)
    ]
    names = ["standard", "OHV", "OPV"]
    colors = ["b", "g", "r"]

    set_up_plt(NEURIPS_FONT_FAMILY, use_tex=False)
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))

    env = gym.make("breeding_gym:BreedingGym",
                   initial_population=DATA_PATH.joinpath("geno.txt"),
                   genetic_map=DATA_PATH.joinpath("genetic_map.txt"),
                   )
    env = GeneticDiversityWrapper(env)
    env = SimplifiedBreedingGym(env, individual_per_gen=individual_per_gen)

    for label, index, color in zip(names, indices, colors):
        buffer_gg = np.empty((trials, num_generations + 1))
        buffer_corr = np.empty((trials, num_generations + 1))
        buffer_g_div = np.empty((trials, num_generations))

        for trial_idx in range(trials):
            pop, info = env.reset(options={"index": index})
            buffer_gg[trial_idx, 0] = env.GEBV.mean()
            buffer_corr[trial_idx, 0] = np.mean(env.corrcoef)

            print("Trial:", trial_idx, flush=True)
            for i in range(num_generations):
                action = {"n_bests": n_bests, "n_crosses": n_crosses}
                pop, r, terminated, truncated, info = env.step(action)
                buffer_gg[trial_idx, i + 1] = env.GEBV.mean()
                buffer_corr[trial_idx, i + 1] = np.mean(env.corrcoef)
                buffer_g_div[trial_idx, i] = np.mean(info["genetic_diversity"])

        gg = np.mean(buffer_gg, axis=0) / env.simulator.max_gebv * 100
        corr = np.mean(buffer_corr, axis=0)
        g_div = np.mean(buffer_g_div, axis=0)

        xticks = np.arange(num_generations + 1)

        axs[0].set_xticks(xticks)
        axs[0].set_title("Genetic Gain (%)")
        axs[0].grid(axis='y')
        axs[0].set_xlabel('Generations [Years]')
        axs[0].plot(xticks, gg - gg[0], label=label)

        axs[1].set_xticks(xticks)
        axs[1].set_title("corrcoef")
        axs[1].grid(axis='y')
        axs[1].set_xlabel('Generations [Years]')
        axs[1].plot(xticks, corr)

        axs[2].set_xticks(xticks)
        axs[2].set_title("Genetic Diversity")
        axs[2].grid(axis='y')
        axs[2].set_xlabel('Generations [Years]')
        axs[2].plot(xticks[1:], g_div)


plt.figlegend(loc='upper right')
plt.savefig(f"figures/compare_indexes_{trials}t.png")
plt.show()
