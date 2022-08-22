from datetime import datetime
import numpy as np

def my_GEBV(Fpro, TMARnp):
    now = datetime.now()

    print("Height")
    print(now.strftime("%H:%M:%S"))
    HeightEFFSp = np.multiply(Fpro.T,TMARnp[:,2])
    now = datetime.now()
    print(now.strftime("%H:%M:%S"))
    HeightGEBVp = np.sum(HeightEFFSp,axis=1)
    now = datetime.now()
    print(now.strftime("%H:%M:%S"))
    del HeightEFFSp
    now = datetime.now()
    print(now.strftime("%H:%M:%S"))


    print("Yield")
    now = datetime.now()
    print(now.strftime("%H:%M:%S"))
    YieldEFFSp = np.multiply(Fpro.T,TMARnp[:,1])
    now = datetime.now()
    print(now.strftime("%H:%M:%S"))
    YieldGEBVp = np.sum(YieldEFFSp,axis=1)
    now = datetime.now()
    print(now.strftime("%H:%M:%S"))
    del YieldEFFSp
    now = datetime.now()
    print(now.strftime("%H:%M:%S"))


    print("Heading")
    now = datetime.now()
    print(now.strftime("%H:%M:%S"))
    HeadingEFFSp = np.multiply(Fpro.T,TMARnp[:,3])
    now = datetime.now()
    print(now.strftime("%H:%M:%S"))
    HeadingGEBVp = np.sum(HeadingEFFSp,axis=1)
    now = datetime.now()
    print(now.strftime("%H:%M:%S"))
    del HeadingEFFSp
    now = datetime.now()
    print(now.strftime("%H:%M:%S"))

    print("TKW")
    now = datetime.now()
    print(now.strftime("%H:%M:%S"))
    TKWEFFSp = np.multiply(Fpro.T,TMARnp[:,4])
    now = datetime.now()
    print(now.strftime("%H:%M:%S"))
    TKWGEBVp = np.sum(TKWEFFSp,axis=1)
    now = datetime.now()
    print(now.strftime("%H:%M:%S"))
    del TKWEFFSp
    now = datetime.now()
    print(now.strftime("%H:%M:%S"))

    print("Zeleny")
    now = datetime.now()
    print(now.strftime("%H:%M:%S"))
    ZelenyEFFSp = np.multiply(Fpro.T,TMARnp[:,5])
    now = datetime.now()
    print(now.strftime("%H:%M:%S"))
    ZelenyGEBVp = np.sum(ZelenyEFFSp,axis=1)
    now = datetime.now()
    print(now.strftime("%H:%M:%S"))
    del ZelenyEFFSp
    now = datetime.now()
    print(now.strftime("%H:%M:%S"))

    YieldINDEXp = YieldGEBVp*2
    HeightINDEXp = -HeightGEBVp
    HeadingINDEXp = np.sqrt(HeadingGEBVp.astype(float)* HeadingGEBVp.astype(float))
    ZelenyINDEXp = ZelenyGEBVp
    TKWINDEXp =TKWGEBVp

    INDEXp = YieldINDEXp + HeightINDEXp + HeadingINDEXp +ZelenyINDEXp + TKWINDEXp

    return INDEXp, YieldINDEXp, HeightINDEXp, HeadingINDEXp, TKWINDEXp, ZelenyINDEXp