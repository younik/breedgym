import numpy as np
from .my_cross import my_cross


def my_stackChromosomes(chromosomes, df, dfnp, progenitorsnp, noffspring):
	Fpro = my_cross(0, chromosomes, df, dfnp, progenitorsnp, noffspring).T
	for x in range(1, len(chromosomes)):
		pro = my_cross(x, chromosomes, df, dfnp, progenitorsnp, noffspring).T
		Fpro = np.concatenate((Fpro,pro))
	return Fpro


