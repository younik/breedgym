import pandas as pd
import os 

def load_data():
    dir_path = os.path.dirname(os.path.realpath(__file__))

    map = pd.read_table(f"{dir_path}/data/map.txt", sep="\t")
    
    parents = pd.read_table(f"{dir_path}/data/geno.txt", low_memory=False)
    parents_name = parents.index
    Tparentsnp = parents.T.to_numpy()

    MAR = pd.read_table(f"{dir_path}/data/marker_effects.txt", sep="\t", index_col="Name")

    return Tparentsnp, parents_name, MAR, map
