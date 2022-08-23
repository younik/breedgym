from datetime import datetime
import numpy as np

def my_GEBV(Fpro, MAR):
    HeightGEBVp = np.dot(Fpro.T, MAR["Height"])
    YieldGEBVp = np.dot(Fpro.T, MAR["Yield"])
    HeadingGEBVp = np.dot(Fpro.T, MAR["Heading"])
    TKWGEBVp = np.dot(Fpro.T, MAR["TKW"])
    ZelenyGEBVp = np.dot(Fpro.T, MAR["Zeleny"])

    YieldINDEXp = YieldGEBVp*2
    HeightINDEXp = -HeightGEBVp
    HeadingINDEXp = np.sqrt(HeadingGEBVp.astype(float)* HeadingGEBVp.astype(float))
    ZelenyINDEXp = ZelenyGEBVp
    TKWINDEXp = TKWGEBVp

    INDEXp = YieldINDEXp + HeightINDEXp + HeadingINDEXp + ZelenyINDEXp + TKWINDEXp

    return INDEXp, YieldINDEXp, HeightINDEXp, HeadingINDEXp, TKWINDEXp, ZelenyINDEXp