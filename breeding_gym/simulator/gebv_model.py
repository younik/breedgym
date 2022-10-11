import numpy as np


class GEBVModel:

    def __init__(
        self,
        marker_effects: np.ndarray,
    ) -> None:
        self.marker_effects = marker_effects

        self.positive_mask = self.marker_effects > 0

        self.max = 2 * self.marker_effects[self.positive_mask].sum(axis=0)
        self.min = 2 * self.marker_effects[~self.positive_mask].sum(axis=0)
        self.mean = self.marker_effects.sum(axis=0)
        # using variance property for sum of independent variables
        self.var = (self.marker_effects**2).mean(axis=0) / 2

    def __call__(self, population: np.ndarray) -> np.ndarray:
        monoploidy = population.sum(axis=-1, dtype=np.int8)
        dot = np.dot(monoploidy, self.marker_effects)
        return dot

    def optimal_haploid_value(self, population):
        positive_mask = self.positive_mask.squeeze()

        optimal_haploid_pop = np.empty(
            (population.shape[0], population.shape[1]), dtype='bool'
        )

        optimal_haploid_pop[:, positive_mask] = np.any(
            population[:, positive_mask],
            axis=-1
        )
        optimal_haploid_pop[:, ~positive_mask] = np.all(
            population[:, ~positive_mask],
            axis=-1
        )

        return 2 * self(optimal_haploid_pop[:, :, None])
