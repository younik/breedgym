import pandas as pd
import numpy as np
from numpy import random
import sys

# use arguments file, nunber of best geneotyes, number of offspring per cross
OUTFILE = "Outfilex.txt"
BS = 10
NOF = 50
SPLIT= 0.2


############################################
############################################
# load the data
############################################
############################################

# this saves the maprents as 'Tparentsnp'
# and the marker effect matrix as 'TMARnp'
exec(open("Scripts/LoadData.py").read())

# GEBV scores are calculated using the matrix Fpro as input
Fpro = Tparentsnp


# calculate the maker effects
# uses  np.multiply(Fpro.T,TMARnp[:,2]) for each marker effect to generate 
# 'INDEXp' is the index score for all individuals
# 'YieldGEBVp' is the yield GEBV
# 'HeightGEBVp' is the Height GEBV
# 'HeadingGEBVp' is the Heading GEBV
# 'TKWGEBVp' is the TKW GEBV
# 'Zeleny' is the Zeleny GEBV
exec(open("Scripts/my_GEBV.py").read())
exec(open("Scripts/my_getGEBV.py").read())
GG = GEBVscores
# calculate the diversity of the crosses
IP = Tparentsnp
cormod=np.corrcoef(IP,rowvar=False)
CC = np.average(cormod[cormod< 1])

# save the original values
INDEX = INDEXp
YieldGEBV = YieldGEBVp
HeightGEBV = HeightGEBVp
HeadingGEBV = HeadingGEBVp
TKWGEBV = TKWGEBVp
ZelenyGEBV = ZelenyGEBVp
Oparentsnp =Tparentsnp
# choose number of offspring per cross
noffspring = int(NOF)

# get a unique list of chromosomes
chromosomes = sorted(list(set(dfnp[:,1])))




##########################################
# save the best ten lines

BEST = int(BS)
print(BEST)

ID = np.array(list(range(len(INDEXp))))
# select the best ten parents
exec(open("Scripts/my_besttenGEBV.py").read())
OO = GEBVbest.T

############################################
############################################
# selection process
############################################
############################################

exec(open("Scripts/my_stackChromosmes.py").read())
exec(open("Scripts/my_cross.py").read())
###########################################
# brute breeding
###########################################


Ngenerations = 10

#exec(open("Scripts/01_selection_BRUTE.py").read())
#exec(open("Scripts/02_simulate_crosses.py").read())
