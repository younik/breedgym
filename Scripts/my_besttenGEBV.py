GE = ID[INDEXp > np.sort(INDEXp)[-(BEST+1)]]
GEBVbest = (INDEXp[GE])
GEBVbest = np.vstack((GEBVbest,YieldINDEXp[GE]))
GEBVbest = np.vstack((GEBVbest,HeightINDEXp[GE]))
GEBVbest = np.vstack((GEBVbest,HeadingINDEXp[GE]))
GEBVbest = np.vstack((GEBVbest,TKWINDEXp[GE]))
GEBVbest = np.vstack((GEBVbest,ZelenyINDEXp[GE]))
GEBVbest = np.vstack((GEBVbest,Childnames[GE]))
