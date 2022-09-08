from breeding_gym.breeding_gym import BreedingGym
import numpy as np


def test_cross_r():
    def const_co_mask(self, n_progenies, chr_idx):
        marker_per_chr = self.recombination_vectors[chr_idx].shape[0]
        return np.ones((n_progenies, marker_per_chr, 2), dtype="int")
    BreedingGym._get_crossover_mask = const_co_mask

    env = BreedingGym()
    init_pop = env.reset()
    p0, p1 = init_pop[0], init_pop[1]
    assert p0.shape == p1.shape

    new_pop, _, _, _, _ = env.step(np.array([[0, 1]]))

    assert new_pop.shape == (1, p0.shape[0], 2)

    ind = new_pop[0]
    assert np.all(ind[:, 0] == p0[:, 1])
    assert np.all(ind[:, 1] == p1[:, 1])


def test_caching():
    env = BreedingGym()

    env.reset(return_info=False)

    GEBV = env.GEBV
    GEBV_copy = np.copy(GEBV)
    GEBV2 = env.GEBV
    assert id(GEBV) == id(GEBV2)
    assert np.all(GEBV_copy == GEBV2)

    corrcoef = env.corrcoef
    corrcoef_copy = np.copy(corrcoef)
    corrcoef2 = env.corrcoef
    assert id(corrcoef) == id(corrcoef2)
    assert np.all(corrcoef_copy == corrcoef2)

    action = np.array([[1, 3], [4, 2]])
    env.step(action)

    GEBV3 = env.GEBV
    corrcoef3 = env.corrcoef
    assert id(corrcoef) != id(corrcoef3)
    assert id(GEBV) != id(GEBV3)
