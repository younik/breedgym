import numpy as np

from .load_data import load_data
from .my_GEBV import my_GEBV
from .my_getGEBV import my_getGEBV
from .my_besttenGEBV import my_besttenGEBV
from .selection_BRUTE import selection_BRUTE
from .simulate_crosses import simulate_crosses

import sys


def main(OUTFILE, BEST, noffspring, SPLIT, seed=None):
	np.random.seed(seed)
	Tparentsnp, Tparentsname, MAR, df = load_data()

	# calculate the maker effects
	INDEXp, YieldINDEXp, HeightINDEXp, HeadingINDEXp, TKWINDEXp, ZelenyINDEXp = my_GEBV(Tparentsnp, MAR)
	GEBVscores = my_getGEBV(INDEXp, YieldINDEXp, HeightINDEXp, HeadingINDEXp, TKWINDEXp, ZelenyINDEXp)

	## GG è solo uno stack per GEBVscores, CC per il coefficiente che dice quanto sono correlati
	GG = GEBVscores
	cormod = np.corrcoef(Tparentsnp,rowvar=False)
	CC = np.average(cormod[cormod < 1])

	Oparentsnp = Tparentsnp
	# choose number of offspring per cross

	# get a unique list of chromosomes
	chromosomes = df['CHR.PHYS'].unique()
	
	# select the best ten parents
	indices = np.arange(len(INDEXp))
	GEBVbest, GE = my_besttenGEBV(indices, INDEXp, BEST, YieldINDEXp, HeightINDEXp, HeadingINDEXp, TKWINDEXp, ZelenyINDEXp, Tparentsname)

	#OO è solo uno stack per GEBVbest, il GEBV dei parents
	OO = GEBVbest.T

	n_generations = 10
	for _ in range(n_generations):
		IP, INDEXp, IPname = selection_BRUTE(SPLIT, BEST, INDEXp, Oparentsnp, Tparentsnp, Tparentsname, GE)
		GG, CC, OO, Tparentsnp, Tparentsname, INDEXp, GE = simulate_crosses(IP, IPname, chromosomes, INDEXp, GEBVbest, MAR, BEST, df, noffspring, GG, CC, OO)
		
	OUT = np.hstack((GG,CC))

	np.savetxt(OUTFILE, OUT)
	np.savetxt(OUTFILE.replace("Outfile","Offspring"), OO, fmt='%s', delimiter=',')


if __name__ == "__main__":
	# use arguments file, nunber of best geneotyes, number of offspring per cross
	OUTFILE = sys.argv[1]
	BEST = sys.argv[2]

	noffspring = sys.argv[3]  # number of offsprings per cross
	SPLIT= sys.argv[4]  # fraction of random genotypes used to produce next gen

	print("OUTFILE = ", OUTFILE)
	print("Best = ", BEST)
	print("NOF = " , noffspring)
	print("Split = ", SPLIT)

	main(OUTFILE, BEST, noffspring, SPLIT)