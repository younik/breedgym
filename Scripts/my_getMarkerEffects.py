HeightEFFS = np.multiply(parents,TMARnp[:,2])
HeightGEBV = np.sum(HeightEFFS,axis=1)
del HeightEFFS

YieldEFFS = np.multiply(parents,TMARnp[:,1])
YieldGEBV = np.sum(YieldEFFS,axis=1)
del YieldEFFS

HeadingEFFS = np.multiply(parents,TMARnp[:,3])
HeadingGEBV = np.sum(HeadingEFFS,axis=1)
del HeadingEFFS

TKWEFFS = np.multiply(parents,TMARnp[:,4])
TKWGEBV = np.sum(TKWEFFS,axis=1)
del TKWEFFS

ZelenyEFFS = np.multiply(parents,TMARnp[:,5])
ZelenyGEBV = np.sum(ZelenyEFFS,axis=1)
del ZelenyEFFS
