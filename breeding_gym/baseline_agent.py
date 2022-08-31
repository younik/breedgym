import numpy as np


def default_f_index(GEBV):
    GEBV_copy = np.copy(GEBV)
    GEBV_copy[:, 2] = np.abs(GEBV_copy[:, 2])

    weights = np.array([2, -1, 1, 1, 1])
    return np.dot(GEBV_copy, weights)


def optimal_haploid_value(env, GEBV):
    positive_marker_mask = default_f_index(env.marker_effects.to_numpy()) > 0
    heterozigosity_mask = env.population == 1
    optimistic_pop = np.copy(env.population)
    optimistic_pop += heterozigosity_mask * positive_marker_mask
    optimistic_pop -= heterozigosity_mask * ~positive_marker_mask

    optimistic_GEBV = env.GEBV(optimistic_pop)
    return default_f_index(optimistic_GEBV)


class BaselineAgent:

    def __init__(self, best=10, n_offspring=50, f_index=default_f_index):
        self.best = best
        self.n_offspring = n_offspring
        self.f_index = f_index

    def __call__(self, GEBV):
        indices = self.f_index(GEBV)

        # retrieve the self.best population indices
        best_pop = np.argpartition(indices, -self.best)[-self.best:]

        mesh1, mesh2 = np.meshgrid(best_pop, best_pop)
        triu_indices = np.triu_indices(self.best, k=1)
        mesh1 = mesh1[triu_indices]
        mesh2 = mesh2[triu_indices]
        action = np.stack([mesh1, mesh2], axis=1)
        action = np.repeat(action, self.n_offspring, axis=0)

        return action