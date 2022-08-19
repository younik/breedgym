# make the first cross, to create children
# where the best ofspring from each cross will be saved
MUM=0
DAD=1
progenitorsnp = IP[:,(MUM,DAD)]
Fpro = my_stackChromosomes()
exec(open("Scripts/my_GEBV.py").read())
Children = Fpro[:,INDEXp == np.max(INDEXp)]
Childnames =np.array(["("+IPname[MUM] +" x "+IPname[DAD]+")"])
# now loop the rest of the crosses
for MUM  in range((IP.shape[1]-1)):
	for DAD in range(MUM+1,IP.shape[1]):
		print("Mum = ",MUM,",DAD = ",DAD)
		if MUM == 0 and DAD ==1: # eliminate the first cross from the loop
		 	break		
		progenitorsnp = IP[:,(MUM,DAD)]
		Fpro = my_stackChromosomes() # make the cross
		exec(open("Scripts/my_GEBV.py").read()) # calculate GEBV
		# only select the best offspirng
		PO = Fpro[:,INDEXp == np.max(INDEXp)]
		# add this offspring to the next generation
		Children =np.concatenate((Children, PO), axis=1)
		Childnames =np.concatenate((Childnames,np.array([("("+IPname[MUM] +" x "+IPname[DAD]+")")])))

########################################
# get scores
Fpro = Children
exec(open("Scripts/my_GEBV.py").read())
exec(open("Scripts/my_getGEBV.py").read())

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
exec(open("Scripts/my_besttenGEBV.py").read())
OO = np.vstack((OO,GEBVbest.T))
