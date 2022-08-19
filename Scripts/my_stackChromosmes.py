def my_stackChromosomes():
	n = progenitorsnp.shape[0]
	Fpro = my_cross(0).T
	for x in range(1, len(chromosomes)):
		pro = my_cross(x).T
		Fpro = np.concatenate((Fpro,pro))
	return(Fpro)


