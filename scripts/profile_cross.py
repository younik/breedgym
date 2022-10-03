import cProfile
from breeding_gym.simulator import BreedingSimulator
import numpy as np
from breeding_gym.utils.paths import DATA_PATH


n_progenies = 300
n_markers = 150_000

size = (n_progenies, 2, n_markers, 2)
parents = np.random.choice(a=[False, True], size=size, p=[0.5, 0.5])
simulator = BreedingSimulator(
    genetic_map=DATA_PATH.joinpath("medium_genetic_map.txt")
)

with cProfile.Profile() as pr:
    simulator.cross(parents)

pr.dump_stats('profile.prof')
