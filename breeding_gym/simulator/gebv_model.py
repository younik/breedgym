import numpy as np
import jax.numpy as jnp
import jax


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

        props = GEBVModel._effect_properties(self.marker_effects)
        self.positive_mask, self.max, self.min, self.mean, self.var = props

    def __call__(self, population: jnp.ndarray) -> jnp.ndarray:
        return GEBVModel._gebv(population, self.marker_effects)

    @jax.jit
    def _gebv(population, marker_effects):
        monoploidy = population.sum(axis=-1, dtype=jnp.int8)
        return jnp.dot(monoploidy, marker_effects)

    @jax.jit
    def _effect_properties(marker_effects):
        positive_mask = marker_effects > 0

        max_ = 2 * jnp.sum(marker_effects, axis=0, where=positive_mask)
        min_ = 2 * jnp.sum(marker_effects, axis=0, where=~positive_mask)
        mean = jnp.sum(marker_effects, axis=0)
        # using variance property for sum of independent variables
        var = jnp.mean(marker_effects**2, axis=0) / 2

        return positive_mask, max_, min_, mean, var

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
