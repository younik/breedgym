import numpy as np
from numpy import random


def selection_BRUTE(SPLIT, BEST, INDEXp, Oparentsnp, Tparentsnp, Tparentsname, GE):
	if float(SPLIT) > 0 and float(SPLIT) < 1:
		NEW = int(BEST*float(SPLIT))
		OLD = int(BEST-(BEST*float(SPLIT)))
		# make an index to slice the data
		ID = np.array(list(range(len(INDEXp))))
		# select the best parents
		GE = ID[INDEXp > np.sort(INDEXp)[-(NEW+1)]]
		# create the dataframe IP which contains parents for diallel crossing
		GO = random.randint(0,(Oparentsnp.shape[1]),OLD)
		IP = Tparentsnp[:,GE]
		IP2 = Oparentsnp[:,GO]
		IP = np.hstack((IP,IP2))
		IPname = Tparentsname[GE]

		return IP, INDEXp, IPname

	elif float(SPLIT) == 0: # if SPLIT is zero then only use old genotypes
		OLD = int(BEST-(BEST*float(SPLIT)))
		# make an index to slice the data
		ID = np.array(list(range(len(INDEXp))))
		# create the dataframe IP which contains parents for diallel crossing
		GO = random.randint(0,(Oparentsnp.shape[1]),OLD)
		IP = Oparentsnp[:,GO]
		IPname = Tparentsname[GE]

		return IP, INDEXp, IPname

	elif float(SPLIT) == 1: # if SPLIT is 1 then only use the latest generation
		NEW = int(BEST*float(SPLIT))
		# make an index to slice the data
		ID = np.array(list(range(len(INDEXp))))
		# select the best parents
		GE = ID[INDEXp > np.sort(INDEXp)[-(NEW+1)]]
		# create the dataframe IP which contains parents for diallel crossing
		IP = Tparentsnp[:,GE]
		IPname = Tparentsname[GE]

		return IP, INDEXp, IPname
