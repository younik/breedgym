import numpy as np
import pulp


def yield_index(env):
    return env.GEBV["Yield"]


def optimal_haploid_pop(env):
    optimal_haploid_pop = np.empty_like(env.population)
    positive_mask = env.simulator.marker_effects > 0

    optimal_haploid_pop[:, positive_mask] = np.max(
        env.population[:, positive_mask], axis=-1
    )[:, :, None]
    optimal_haploid_pop[:, ~positive_mask] = np.min(
        env.population[:, ~positive_mask], axis=-1
    )[:, :, None]

    return optimal_haploid_pop


def optimal_haploid_value(env):
    oh_pop = optimal_haploid_pop(env)
    return env.simulator.GEBV(oh_pop)["Yield"]


def optimal_population_value(env):
    def my_dot(a, b):
        assert len(a) == len(b)
        return pulp.lpSum([a[i] * b[i] for i in range(len(a))])

    problem = pulp.LpProblem("Optimal Population Value", pulp.LpMaximize)
    binary = {"lowBound": 0, "upBound": 1, "cat": pulp.LpInteger}

    c = env.simulator.marker_effects
    n_markers = len(c)
    best_set = pulp.LpVariable.dicts("best_set", range(n_markers), **binary)
    problem += pulp.lpDot(c, best_set)

    x = pulp.LpVariable.dicts("x", range(len(env.population)), **binary)
    problem += pulp.lpSum(x) == 20

    G = optimal_haploid_pop(env)

    for marker_idx in range(n_markers):
        # y <= G \cdot x
        problem += best_set[marker_idx] <= my_dot(G[:, marker_idx], x)

        # y >= 1 - (1 - G) \cdot x
        not_G = 1 - G[:, marker_idx]
        problem += best_set[marker_idx] >= 1 - my_dot(not_G, x)

    res = problem.solve()
    return res.x  # it doesn't return the OPV per ind but the selection mask
