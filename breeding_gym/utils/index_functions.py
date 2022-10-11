import numpy as np


def yield_index(env):
    return env.GEBV["Yield"]


def optimal_haploid_value(env):
    GEBV_model = env.simulator.GEBV_model
    return GEBV_model.optimal_haploid_value(env.population).squeeze()


def optimal_haploid_pop(env):
    optimal_haploid_pop = np.empty(
        (env.population.shape[0], env.population.shape[1]), dtype='bool'
    )

    positive_mask = env.simulator.GEBV_model.positive_mask.squeeze()

    optimal_haploid_pop[:, positive_mask] = np.logical_or(
        env.population[:, positive_mask, 0],
        env.population[:, positive_mask, 1]
    )
    optimal_haploid_pop[:, ~positive_mask] = np.logical_and(
        env.population[:, ~positive_mask, 0],
        env.population[:, ~positive_mask, 1]
    )

    return optimal_haploid_pop


def optimal_population_value(n):

    def optimal_population_value_f(env):
        output = np.zeros(len(env.population), dtype='bool')
        GEBV_model = env.simulator.GEBV_model
        positive_mask = GEBV_model.marker_effects[:, 0] > 0
        current_set = ~positive_mask
        G = optimal_haploid_pop(env)

        for _ in range(n):
            G[:, positive_mask] = np.logical_or(
                G[:, positive_mask], current_set[positive_mask]
            )
            G[:, ~positive_mask] = np.logical_and(
                G[:, ~positive_mask], current_set[~positive_mask]
            )

            best_idx = np.argmax(env.simulator.GEBV(G[:, :, None]))
            output[best_idx] = True
            current_set = G[best_idx]
            G[best_idx] = ~positive_mask  # "remove" it

        assert np.count_nonzero(output) == n
        return output  # not OPV but a mask

    return optimal_population_value_f
