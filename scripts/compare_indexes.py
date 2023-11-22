import time

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

from breedgym.simulator.gebv_model import GEBVModel
from breedgym.utils.index_functions import (
    optimal_haploid_value,
    optimal_population_value,
    yield_index,
)
from breedgym.utils.paths import DATA_PATH
from breedgym.utils.plot_utils import NEURIPS_FONT_FAMILY, set_up_plt
from breedgym.wrappers import SimplifiedBreedGym


class GeneticDiversityWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        genetic_diversity = self.genetic_diversity(
            self.simulator.GEBV_model, obs, mrk_axis=1
        )
        info["genetic_diversity"] = genetic_diversity
        return obs, info

    def step(self, action):
        parents = self.population[action]
        genetic_diversity = self.genetic_diversity(
            self.simulator.GEBV_model, parents, mrk_axis=2
        )
        obs, r, ter, tru, info = super().step(action)
        info["genetic_diversity"] = genetic_diversity
        return obs, r, ter, tru, info

    def genetic_diversity(self, GEBV_model, population, mrk_axis=1):
        other_axis = (i for i in range(len(population.shape)) if i != mrk_axis)
        population = population.transpose(mrk_axis, *other_axis)
        population = population.reshape(1, population.shape[0], -1)
        max_GEBV = optimal_haploid_value(GEBV_model)(population)
        neg_GEBV = GEBVModel(-GEBV_model.marker_effects)
        min_GEBV = -optimal_haploid_value(neg_GEBV)(population)
        return (max_GEBV - min_GEBV).squeeze()


if __name__ == "__main__":
    num_generations = 10
    individual_per_gen = 200
    n_bests = 20
    n_crosses = 10

    trials = 50
    names = ["standard"]  # , "OHV"]#, "OPV"]
    colors = ["b", "g", "r"]

    set_up_plt(NEURIPS_FONT_FAMILY, use_tex=False)
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))

    env = gym.make(
        "breedgym:BreedGym",
        initial_population=DATA_PATH.joinpath("sample_full_pop_geno.txt"),
        genetic_map=DATA_PATH.joinpath("sample_with_r_genetic_map.txt"),
    )
    env = GeneticDiversityWrapper(env)
    env = SimplifiedBreedGym(env, individual_per_gen=individual_per_gen)

    GEBV_model = env.simulator.GEBV_model
    indices = [
        yield_index(GEBV_model),
        optimal_haploid_value(GEBV_model, F=0.7, B=12, chr_lens=env.simulator.chr_lens),
        optimal_population_value(
            GEBV_model, n_bests, F=0.4, B=12, chr_lens=env.simulator.chr_lens
        ),
    ]

    start_time = time.time()
    for label, index, color in zip(names, indices, colors):
        buffer_gg = np.empty((trials, num_generations + 1))
        buffer_g_div = np.empty((trials, num_generations + 1))

        for trial_idx in range(trials):
            obs, info = env.reset(options={"index": index})
            buffer_gg[trial_idx, 0] = env.GEBV.mean()
            buffer_g_div[trial_idx, 0] = info["genetic_diversity"]

            print(f"{label}, trial {trial_idx}", flush=True)
            for i in range(num_generations):
                action = {"n_bests": n_bests, "n_crosses": n_crosses}
                obs, r, terminated, truncated, info = env.step(action)
                buffer_gg[trial_idx, i + 1] = env.GEBV.mean()
                buffer_g_div[trial_idx, i + 1] = info["genetic_diversity"]

        buffer_gg = (
            (buffer_gg - buffer_gg[:, 0][:, None]) / env.simulator.max_gebv * 100
        )
        gg = np.mean(buffer_gg, axis=0)
        gg_stderr = np.std(buffer_gg, axis=0, ddof=1) / np.sqrt(trials)
        buffer_g_div /= env.simulator.max_gebv - env.simulator.min_gebv
        buffer_g_div *= 100
        g_div = np.mean(buffer_g_div, axis=0)
        g_div_stderr = np.std(buffer_g_div, axis=0, ddof=1) / np.sqrt(trials)

        xticks = np.arange(num_generations + 1)

        axs[0].set_xticks(xticks)
        axs[0].set_title("Genetic Gain (%)")
        axs[0].grid(axis="y")
        axs[0].set_xlabel("Generations [Years]")
        axs[0].plot(xticks, gg, label=label)
        axs[0].fill_between(xticks, gg - gg_stderr, gg + gg_stderr, alpha=0.2)

        axs[1].set_xticks(xticks)
        axs[1].set_title("Genetic Diversity")
        axs[1].grid(axis="y")
        axs[1].set_xlabel("Generations [Years]")
        axs[1].plot(xticks, g_div)
        axs[1].fill_between(
            xticks, g_div - g_div_stderr, g_div + g_div_stderr, alpha=0.2
        )

    print("Elapsed time", time.time() - start_time, flush=True)
    plt.figlegend(loc="upper right")
    plt.savefig("figures/test_baseline.png")
    plt.show()
