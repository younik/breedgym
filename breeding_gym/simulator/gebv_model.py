import numpy as np
import jax.numpy as jnp


class GEBVModel:

    def __init__(
        self,
        marker_effects: np.ndarray,
    ) -> None:
        self.marker_effects = marker_effects

        positive_mask = self.marker_effects > 0

        self.max = 2 * self.marker_effects[positive_mask].sum(axis=0)
        self.min = 2 * self.marker_effects[~positive_mask].sum(axis=0)

        self.mean = self.marker_effects.sum(axis=0)
        # using variance property for sum of independent variables
        self.var = (self.marker_effects**2).mean(axis=0) / 2

    def __call__(self, population: np.ndarray) -> np.ndarray:
        monoploidy = population.sum(axis=-1)
        return jnp.dot(monoploidy, self.marker_effects)
