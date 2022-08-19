# load the data
df = pd.read_table("Data/PyMap.txt",sep="\t")
parents = pd.read_table("Data/PyGeno.txt",low_memory=False)

Tparents = parents.T
# convert to numpy arrays for easy indexing
Tparentsnp = Tparents.to_numpy()
dfnp = df.to_numpy()

# load the marker effects
MAR = pd.read_table("Data/AllMarkerEffects.txt",sep="\t")
TMAR = MAR.T
TMARnp = MAR.to_numpy()
