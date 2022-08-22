YieldINDEX = YieldGEBV*2
HeightINDEX = -HeightGEBV
HeadingINDEX = -np.sqrt(HeadingGEBV *HeadingGEBV)
ZelenyINDEX = ZelenyGEBV
TKWINDEX =TKWGEBV

# get the index of the lines
INDEX = YieldINDEX + HeightINDEX + HeadingINDEX +ZelenyINDEX + TKWINDEX
