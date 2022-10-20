import numpy as np


def yield_index(GEBV_model):
    def yield_index_f(pop):
        return GEBV_model(pop)[:, 0]

    return yield_index_f


def optimal_haploid_value(GEBV_model):
    def optimal_haploid_value_f(pop):
        return GEBV_model.optimal_haploid_value(pop).squeeze()

    return optimal_haploid_value_f


def optimal_haploid_pop(GEBV_model, population):
    optimal_haploid_pop = np.empty(
        (population.shape[0], population.shape[1]), dtype='bool'
    )

    positive_mask = GEBV_model.positive_mask.squeeze()

    optimal_haploid_pop[:, positive_mask] = np.logical_or(
        population[:, positive_mask, 0],
        population[:, positive_mask, 1]
    )
    optimal_haploid_pop[:, ~positive_mask] = np.logical_and(
        population[:, ~positive_mask, 0],
        population[:, ~positive_mask, 1]
    )

    return optimal_haploid_pop


def optimal_population_value(GEBV_model, n):

    def optimal_population_value_f(population):
        output = np.zeros(len(population), dtype='bool')
        positive_mask = GEBV_model.marker_effects[:, 0] > 0
        current_set = ~positive_mask
        G = optimal_haploid_pop(GEBV_model, population)

        for _ in range(n):
            G[:, positive_mask] = np.logical_or(
                G[:, positive_mask], current_set[positive_mask]
            )
            G[:, ~positive_mask] = np.logical_and(
                G[:, ~positive_mask], current_set[~positive_mask]
            )

            best_idx = np.argmax(GEBV_model(G[:, :, None]))
            output[best_idx] = True
            current_set = G[best_idx]
            G[best_idx] = ~positive_mask  # "remove" it

        assert np.count_nonzero(output) == n
        return output  # not OPV but a mask

    return optimal_population_value_f
