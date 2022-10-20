from breeding_gym.simulator import BreedingSimulator
import pytest
import numpy as np
from breeding_gym.utils.paths import DATA_PATH
import pandas as pd


class MockSimulator(BreedingSimulator):

    def __init__(
        self,
        marker_effects = None,
        recombination_vec = None,
        n_chr=10
    ):
        if marker_effects is not None:
            n_markers = len(marker_effects)
        elif recombination_vec is not None:
            n_markers = len(recombination_vec)
        else:
            raise Exception("You must specify at least one between marker_effects and recombination_vec")
        
        if marker_effects is None:
            marker_effects = np.random.randn(n_markers)
        if recombination_vec is None:
            recombination_vec = np.random.uniform(size=n_markers)
            recombination_vec /= n_markers / 20

        if len(marker_effects) != len(recombination_vec):
            raise Exception("marker_effects and recombination_vec must have same length")

        chromosomes = np.arange(n_markers) // (n_markers // n_chr)

        data = np.vstack([chromosomes, recombination_vec, marker_effects]).T
        genetic_map = pd.DataFrame(data, columns=["Chr", "RecombRate", "Effect"])

        super().__init__(genetic_map=genetic_map)
        self.recombination_vec = recombination_vec
        

@pytest.mark.parametrize("idx", [0, 1])
def test_cross_r(idx):
    n_markers = 1000
    recombination_vec = np.zeros(n_markers, dtype="bool")
    recombination_vec[0] = idx
    simulator = MockSimulator(recombination_vec=recombination_vec)

    size = (1, 2, simulator.n_markers, 2)
    parents = np.random.choice(a=[False, True], size=size, p=[0.5, 0.5])

    new_pop = simulator.cross(parents)

    assert new_pop.shape == (1, simulator.n_markers, 2)

    ind = new_pop[0]
    assert np.all(ind[:, 0] == parents[0, 0, :, idx])
    assert np.all(ind[:, 1] == parents[0, 1, :, idx])


def test_equal_parents():
    simulator = BreedingSimulator(
        genetic_map=DATA_PATH.joinpath("small_genetic_map.txt")
    )

    parents = np.zeros((1, 2, simulator.n_markers, 2), dtype="bool")
    child = simulator.cross(parents)
    assert np.all(child == 0)

    parents = np.ones((1, 2, simulator.n_markers, 2), dtype="bool")
    child = simulator.cross(parents)
    assert np.all(child == 1)


def test_ad_hoc_cross():
    parent_0 = np.array([
        [0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0],
        [1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1]
    ], dtype='bool')
    parent_1 = np.array([
        [0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0],
        [0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0]
    ], dtype='bool')
    assert parent_0.shape == parent_1.shape

    rec_vec = np.array(
        [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
        dtype=np.int8
    )
    assert len(rec_vec) == parent_1.shape[1]
    simulator = MockSimulator(recombination_vec=rec_vec)

    parents = np.empty((1, 2, *parent_0.T.shape), dtype='bool')
    parents[0, 0] = parent_0.T
    parents[0, 1] = parent_1.T
    child = simulator.cross(parents)

    assert child.shape == (1, *parent_0.T.shape)

    chr_idx = 0
    for mrk_idx, rec_prob in enumerate(rec_vec):
        if rec_prob == 1:
            chr_idx = 1 - chr_idx
        assert child[1, mrk_idx, 0] == parent_0[chr_idx, mrk_idx]
        assert child[1, mrk_idx, 1] == parent_1[chr_idx, mrk_idx]

def test_phenotyping():
    simulator = BreedingSimulator(
        genetic_map=DATA_PATH.joinpath("small_genetic_map.txt"),
        h2=[0.5]
    )

    size = (10, simulator.n_markers, 2)
    pop = np.random.choice(a=[False, True], size=size, p=[0.5, 0.5])

    assert np.any(simulator.phenotype(pop) != simulator.GEBV(pop))

    simulator.h2 = np.array([1])
    assert np.all(simulator.phenotype(pop) == simulator.GEBV(pop))
