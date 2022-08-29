import numpy as np


def default_f_index(GEBV):
    return 2 * GEBV[:, 0] - GEBV[:, 1] + np.abs(GEBV[:, 2]) + GEBV[:, 3] + GEBV[:, 4]

class BaselineAgent:

    def __init__(self, best=10, n_offspring=50, f_index=default_f_index) -> None:
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