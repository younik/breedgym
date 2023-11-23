import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax

from breedgym.simulator.simulator import BreedingSimulator
from breedgym.utils.paths import DATA_PATH


def breeding_func(population, params, key):
    current_pop = population
    for current_w in params.values():
        current_pop = diff_cross(current_pop, current_w, key)

    monoploidy = current_pop.sum(axis=-1)
    return jnp.dot(monoploidy, marker_effects).mean()


def loss(population, params, key):
    return -breeding_func(population, params, key)


def normalize_params(params):
    for k in params.keys():
        params[k] /= np.linalg.norm(params[k], axis=1, keepdims=True, ord=1)


if __name__ == "__main__":
    budgets = [30, 20, 10, 5]
    population = np.random.choice(
        a=[0.0, 1.0], size=(budgets[0], 10_000, 2), p=[0.5, 0.5]
    )

    simulator = BreedingSimulator(
        genetic_map=DATA_PATH.joinpath("small_genetic_map.txt")
    )

    marker_effects = simulator.GEBV_model.marker_effects
    diff_cross = simulator.differentiable_cross_func

    d_loss = jax.grad(loss, argnums=1)

    params = {}
    for gen in range(1, len(budgets)):
        gen_w = np.random.rand(budgets[gen], budgets[gen - 1], 2)
        params[f"w_{gen}"] = gen_w
    normalize_params(params)

    n_steps = int(5e2)
    lr = 1e-3
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)
    values = np.empty(n_steps)

    for i in range(n_steps):
        print(i, flush=True)
        key = jax.random.PRNGKey(np.random.randint(2**32))
        grads = d_loss(population, params, key)

        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        normalize_params(params)

        key2 = jax.random.PRNGKey(np.random.randint(2**32))
        values[i] = breeding_func(population, params, key2)

    plt.plot(np.arange(n_steps), values)
    plt.savefig("test.png")
