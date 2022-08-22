import numpy as np

from .load_data import load_data
from .my_GEBV import my_GEBV
from .my_getGEBV import my_getGEBV
from .my_besttenGEBV import my_besttenGEBV
from .selection_BRUTE import selection_BRUTE
from .simulate_crosses import simulate_crosses

import sys


def main(OUTFILE, BS, NOF, SPLIT):
	############################################
	############################################
	# load the data
	############################################
	############################################

	# this saves the maprents as 'Tparentsnp'
	# and the marker effect matrix as 'TMARnp'
	Tparents, Tparentsnp, TMARnp, df, dfnp = load_data()

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
	INDEXp, YieldINDEXp, HeightINDEXp, HeadingINDEXp, TKWINDEXp, ZelenyINDEXp = my_GEBV(Fpro, TMARnp)
	GEBVscores = my_getGEBV(INDEXp, YieldINDEXp, HeightINDEXp, HeadingINDEXp, TKWINDEXp, ZelenyINDEXp)

	GG = GEBVscores
	# calculate the diversity of the crosses
	IP = Tparentsnp
	cormod=np.corrcoef(IP,rowvar=False)
	CC = np.average(cormod[cormod< 1])

	# save the original values
	INDEX = INDEXp
	# YieldGEBV = YieldGEBVp
	# HeightGEBV = HeightGEBVp
	# HeadingGEBV = HeadingGEBVp
	# TKWGEBV = TKWGEBVp
	# ZelenyGEBV = ZelenyGEBVp
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
	Tparentsname = Tparents.columns.values
	Childnames=Tparentsname
	GEBVbest = my_besttenGEBV(ID, INDEXp, BEST, YieldINDEXp, HeightINDEXp, HeadingINDEXp, TKWINDEXp, ZelenyINDEXp, Childnames)

	OO = GEBVbest.T


	############################################
	############################################
	# selection process
	############################################
	############################################

	# /!\ not executed in the original code, serves only for import
	# my_stackChromosomes() ##  
	# my_cross()

	###########################################
	# brute breeding
	###########################################

	n_generations = 10
	for _ in range(n_generations):
		IP, INDEXp, IPname = selection_BRUTE(SPLIT, BEST, INDEXp, Oparentsnp, Tparentsnp, Tparentsname)
		GG, CC, OO, Tparentsnp, Tparentsname = simulate_crosses(IP, IPname, chromosomes, INDEXp, GEBVscores, GEBVbest, TMARnp, BEST, df, dfnp, noffspring)
		
	OUT = np.hstack((GG,CC))

	np.savetxt(OUTFILE, OUT)
	np.savetxt(OUTFILE.replace("Outfile","Offspring"), OO, fmt='%s')


if __name__ == "__main__":
	# use arguments file, nunber of best geneotyes, number of offspring per cross
	OUTFILE = sys.argv[1]
	BS = sys.argv[2] # breeding cycles

	NOF = sys.argv[3]  # number of offsprings per cross
	SPLIT= sys.argv[4]  # fraction of random genotypes used to produce next gen

	print("OUTFILE = ",OUTFILE)
	print("Best = ",BS)
	print("NOF = " ,NOF)
	print("Split = ",SPLIT)

	main(OUTFILE, BS, NOF, SPLIT)