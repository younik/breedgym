import timeit

import numpy as np
import pandas as pd

from breedgym.simulator import BreedingSimulator
from breedgym.utils.paths import DATA_PATH


if __name__ == "__main__":
    n_progenies = [300]
    data_sizes = ["big"]

    n_markers = {"small": 10_000, "medium": 150_000, "big": 1406757}

    dataset = {
        "small": "small_genetic_map.txt",
        "medium": "medium_genetic_map.txt",
        "big": "genetic_map.txt",
    }

    repeats = 25

    table = np.empty((len(n_progenies), len(data_sizes)))
    for row, n_prog in enumerate(n_progenies):
        for col, dsize in enumerate(data_sizes):
            size = (n_prog, 2, n_markers[dsize], 2)
            # parents = np.random.choice(a=[False, True], size=size, p=[0.5, 0.5])
            parents = np.zeros(size, dtype="bool")

            simulator = BreedingSimulator(
                genetic_map=DATA_PATH.joinpath(dataset[dsize])
            )

            t = timeit.timeit(
                lambda: simulator.cross(parents).block_until_ready(), number=repeats
            )

            table[row, col] = t / repeats

    df = pd.DataFrame(table, index=n_progenies, columns=data_sizes)
    print(df, flush=True)
