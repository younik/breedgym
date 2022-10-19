import cProfile
from breeding_gym.simulator import BreedingSimulator
import numpy as np
from breeding_gym.utils.paths import DATA_PATH
import timeit


# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = ""

n_progenies = 100
n_markers = 1406757

size = (n_progenies, 2, n_markers, 2)
parents = np.random.choice(a=[False, True], size=size, p=[0.5, 0.5])

simulator = BreedingSimulator(
    genetic_map=DATA_PATH.joinpath("genetic_map.txt")
)

repeats = 100
t = timeit.timeit(lambda : simulator.cross(parents), number=repeats) / repeats
print(t)