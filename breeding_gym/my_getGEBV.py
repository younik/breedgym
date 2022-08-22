import numpy as np


def my_getGEBV(INDEXp, YieldINDEXp, HeightINDEXp, HeadingINDEXp, TKWINDEXp, ZelenyINDEXp):
    GEBVscores = np.average(INDEXp)
    GEBVscores = np.append(GEBVscores,np.average(YieldINDEXp))
    GEBVscores = np.append(GEBVscores,np.average(HeightINDEXp))
    GEBVscores = np.append(GEBVscores,np.average(HeadingINDEXp))
    GEBVscores = np.append(GEBVscores,np.average(TKWINDEXp))
    GEBVscores = np.append(GEBVscores,np.average(ZelenyINDEXp))

    return GEBVscores
