import numpy as np
import jax.numpy as jnp
import jax


@jax.jit
def _gebv(population, marker_effects):
    monoploidy = population.sum(axis=-1, dtype=jnp.int8)
    return jnp.dot(monoploidy, marker_effects)


class GEBVModel:

    def __init__(
        self,
        marker_effects: jnp.ndarray,
        device=None
    ) -> None:
        self.device = device
        self.marker_effects = jax.device_put(
            marker_effects,
            device=self.device
        )
        self.n_traits = marker_effects.shape[1]

        self.positive_mask = self.marker_effects > 0

        self.max = 2 * self.marker_effects[self.positive_mask].sum(axis=0)
        self.min = 2 * self.marker_effects[~self.positive_mask].sum(axis=0)
        self.mean = self.marker_effects.sum(axis=0)
        # using variance property for sum of independent variables
        self.var = (self.marker_effects**2).mean(axis=0) / 2

    def __call__(self, population: jnp.ndarray) -> jnp.ndarray:
        return _gebv(population, self.marker_effects)

    def optimal_haploid_value(self, population):
        if self.n_traits != 1:
            raise ValueError("OHV works only with single trait")
        positive_mask = self.positive_mask.squeeze()

        optimal_haploid_pop = np.empty(population.shape[:-1], dtype='bool')

        optimal_haploid_pop[:, positive_mask] = np.any(
            population[:, positive_mask],
            axis=-1
        )
        optimal_haploid_pop[:, ~positive_mask] = np.all(
            population[:, ~positive_mask],
            axis=-1
        )

        return 2 * self(optimal_haploid_pop[..., None])
