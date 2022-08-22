import numpy as np

def my_cross(IN, chromosomes, df, dfnp, progenitorsnp, noffspring):
	# IN: chromosome number
	chrsize = np.amax(dfnp[df['CHR.PHYS'] == chromosomes[IN],2])
	chrs = dfnp[df['CHR.PHYS'] == chromosomes[IN],2].shape[0]
	#generate parents
	a = progenitorsnp[df['CHR.PHYS'] == chromosomes[IN],:]
	aT = a.T
	# now generate break points
	bp = np.random.rand(noffspring*2)*chrsize
	# split the results between two
	bp = np.reshape(bp, (2, noffspring))
	# assign secondary break points less than the first as 0
	bp[1,bp[1,] < bp[0,] ] =0
	bp[1,bp[1,]-bp[0,] <20] =0
	#assign a starting parent
	pr = np.random.randint(2,size=noffspring)
	# do the reverse
	pr2 = 1 - pr
	# now assign the starting parents to the offspring
	progeny = a[:,pr].T
	rec = dfnp[df['CHR.PHYS'] == chromosomes[IN],2]
	ord = list(range(rec.shape[0]))
	ord = np.array(ord)
	map = np.column_stack((ord,rec))
	for x in range(noffspring):
		m = map[:,1] > bp[0,x]
		progeny[x,m] = aT[pr2[x],m]
		if bp[1,x] > 0:
			m = map[:,1] > bp[1,x]
			progeny[x,m] = aT[pr[x],m]
		
	return progeny
	
	

