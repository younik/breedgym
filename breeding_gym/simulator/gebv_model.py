import jax.numpy as jnp
import pandas as pd

class GEBVModel:

    def __init__(
        self,
        marker_effects: jnp.ndarray,
        trait_names: list[str]
    ) -> None:
        assert len(trait_names) == marker_effects.shape[1]
        
        self.marker_effects = marker_effects
        self.trait_names = trait_names
        
        positive_mask = self.marker_effects > 0
        self.max = 2 * self.marker_effects[positive_mask].sum(axis=0)
        self.min = 2 * self.marker_effects[~positive_mask].sum(axis=0)
        
    def __call__(self, population):
        monoploidy = population.sum(axis=-1)
        GEBV = jnp.dot(monoploidy, self.marker_effects)
        return pd.DataFrame(GEBV, columns=self.trait_names)
