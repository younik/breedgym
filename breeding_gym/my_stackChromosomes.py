import numpy as np
from .my_cross import my_cross


def my_stackChromosomes(chromosomes, df, progenitorsnp, noffspring):
	Fpro = []
	for chr in chromosomes:
		pro = my_cross(chr, df, progenitorsnp, noffspring).T
		Fpro.append(pro)
	return np.concatenate(Fpro)


