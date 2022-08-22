import pandas as pd
import os 

def load_data():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    # load the data
    df = pd.read_table(f"{dir_path}/data/PyMap.txt", sep="\t")
    parents = pd.read_table(f"{dir_path}/data/PyGeno.txt", low_memory=False)

    Tparents = parents.T
    # convert to numpy arrays for easy indexing
    Tparentsnp = Tparents.to_numpy()
    dfnp = df.to_numpy()

    # load the marker effects
    MAR = pd.read_table(f"{dir_path}/data/AllMarkerEffects.txt", sep="\t")
    TMAR = MAR.T
    TMARnp = MAR.to_numpy()

    return Tparents, Tparentsnp, TMARnp, df, dfnp
