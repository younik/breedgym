import numpy as np
from .my_stackChromosomes import my_stackChromosomes
from .my_GEBV import my_GEBV
from .my_getGEBV import my_getGEBV
from .my_besttenGEBV import my_besttenGEBV


def simulate_crosses(IP, IPname, chromosomes, INDEXp, GEBVbest, MAR, BEST, df, noffspring, GG, CC, OO):
	# make the first cross, to create children
	# where the best ofspring from each cross will be saved
	MUM=0
	DAD=1
	progenitorsnp = IP[:,(MUM,DAD)]
	Fpro = my_stackChromosomes(chromosomes, df, progenitorsnp, noffspring)
	INDEXp, YieldINDEXp, HeightINDEXp, HeadingINDEXp, TKWINDEXp, ZelenyINDEXp = my_GEBV(Fpro, MAR)
	Children = Fpro[:,INDEXp == np.max(INDEXp)]
	Childnames = np.array(["("+IPname[MUM] +" x "+IPname[DAD]+")"])
	# now loop the rest of the crosses
	for MUM  in range(IP.shape[1] - 1):
		for DAD in range(MUM + 1, IP.shape[1]):
			print("Mum = ",MUM,",DAD = ",DAD)
			if MUM == 0 and DAD ==1: # eliminate the first cross from the loop
				break		
			progenitorsnp = IP[:,(MUM,DAD)]
			Fpro = my_stackChromosomes(chromosomes, df, progenitorsnp, noffspring) # make the cross
			INDEXp, YieldINDEXp, HeightINDEXp, HeadingINDEXp, TKWINDEXp, ZelenyINDEXp = my_GEBV(Fpro, MAR) # calculate GEBV
			# only select the best offspirng
			PO = Fpro[:,INDEXp == np.max(INDEXp)]
			# add this offspring to the next generation
			Children = np.concatenate((Children, PO), axis=1)
			Childnames = np.concatenate((Childnames,np.array([("("+IPname[MUM] +" x "+IPname[DAD]+")")])))

	########################################
	# get scores
	Fpro = Children

	INDEXp, YieldINDEXp, HeightINDEXp, HeadingINDEXp, TKWINDEXp, ZelenyINDEXp = my_GEBV(Fpro, MAR)
	GEBVscores = my_getGEBV(INDEXp, YieldINDEXp, HeightINDEXp, HeadingINDEXp, TKWINDEXp, ZelenyINDEXp)

	# add the data to the existing data
	GG = np.vstack((GG,GEBVscores))

	# get the correlation between genitors
	cormod=np.corrcoef(IP,rowvar=False)
	CC = np.vstack((CC,np.average(cormod[cormod< 1])))
	# change the parental data to the new offspring
	Tparentsnp=Fpro
	IPname=Childnames
	Tparentsname=Childnames

	ID = np.array(list(range(len(INDEXp))))
	GEBVbest, GE = my_besttenGEBV(ID, INDEXp, BEST, YieldINDEXp, HeightINDEXp, HeadingINDEXp, TKWINDEXp, ZelenyINDEXp, Childnames)
	OO = np.vstack((OO,GEBVbest.T))

	return GG, CC, OO, Tparentsnp, Tparentsname, INDEXp, GE
