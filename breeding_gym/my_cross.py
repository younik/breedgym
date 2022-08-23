import numpy as np

def my_cross(chr, df, progenitorsnp, noffspring):
	pred = df[df['CHR.PHYS'] == chr]['pred']
	chrsize = pred.max()

	a = progenitorsnp[df['CHR.PHYS'] == chr].T
	
	# now generate break points
	bp = np.random.rand(2, noffspring) * chrsize
	bp[1, bp[1] < bp[0]] = 0
	bp[1, bp[1]-bp[0] < 20] = 0

	#assign a starting parent
	pr = np.random.randint(2, size=noffspring)
	
	# now assign the starting parents to the offspring
	progeny = a[pr]
	for x in range(noffspring):
		m = pred > bp[0,x]
		progeny[x, m] = a[1 - pr[x], m] #dopo il primo bp, ci mette il secondo parente
		if bp[1, x] > 0:
			m = pred > bp[1, x]
			progeny[x,m] = a[pr[x], m] #dopo il secondo bp, ci rimette il primo, così secondo solo parte interna
		
	return progeny
	
	

