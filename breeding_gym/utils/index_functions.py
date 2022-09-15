import numpy as np


def yield_index(env):
    return env.GEBV["Yield"]


def optimal_haploid_value(env):
    optimal_haploid_pop = np.empty_like(env.population)
    positive_mask = env.simulator.marker_effects > 0

    optimal_haploid_pop[:, positive_mask] = np.max(
        env.population[:, positive_mask], axis=-1
    )[:, :, None]
    optimal_haploid_pop[:, ~positive_mask] = np.min(
        env.population[:, ~positive_mask], axis=-1
    )[:, :, None]

    return env.simulator.GEBV(optimal_haploid_pop)["Yield"]


def optimal_population_value(n):
    def optimal_population_value_f(env):
        pass

    return optimal_population_value_f
