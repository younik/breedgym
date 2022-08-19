fi <- list.files()
fi <- fi[grep("Offspring",fi)]
o2 <- NA


getdata <- function(INFILE)
{
data <- read.table(INFILE,sep="\t")
XL <- NA
for(LINE in 101:110)
{
d <- data$V1[LINE]
d <- gsub(paste(strsplit(d," ")[[1]][1:6],collapse=" "),"",d)
d <-  gsub("[(]","",d)
d <-  gsub("[)]","",d)
d <-  gsub(" ","",d)
els <- strsplit(d, "x")

els <- els[[1]]
out <- c()

for(x in seq(1,length(els),by=2))
 {out <- rbind(out,(c(els[x], els[x+1])))}
 table(out)
 
for(x in seq(1,length(els),by=4))
 {out <- rbind(out,(c(paste(els[x],els[x+1],sep=" x "), paste(els[x+2],els[x+3],sep=" x "))))}
 table(out)
 
 for(x in seq(1,length(els),by=8))
 {out <- rbind(out,(c(c(paste(els[x:(x+3)],collapse=" x "), paste(els[(x+4):(x+7)],collapse=" x ")))))}
 table(out)
 
 
 for(x in seq(1,length(els),by=16))
 {out <- rbind(out,(c(c(paste(els[x:(x+7)],collapse=" x "), paste(els[(x+8):(x+15)],collapse=" x ")))))}
 table(out)
 
  for(x in seq(1,length(els),by=32))
 {out <- rbind(out,(c(c(paste(els[x:(x+15)],collapse=" x "), paste(els[(x+16):(x+31)],collapse=" x ")))))}
 table(out)
 
  for(x in seq(1,length(els),by=64))
 {out <- rbind(out,(c(c(paste(els[x:(x+31)],collapse=" x "), paste(els[(x+32):(x+63)],collapse=" x ")))))}
 table(out)
 
  for(x in seq(1,length(els),by=128))
 {out <- rbind(out,(c(c(paste(els[x:(x+63)],collapse=" x "), paste(els[(x+64):(x+127)],collapse=" x ")))))}
 table(out)
 
  for(x in seq(1,length(els),by=256))
 {out <- rbind(out,(c(c(paste(els[x:(x+127)],collapse=" x "), paste(els[(x+128):(x+255)],collapse=" x ")))))}
 table(out)
 
 
  for(x in seq(1,length(els),by=512))
 {out <- rbind(out,(c(c(paste(els[x:(x+255)],collapse=" x "), paste(els[(x+256):(x+511)],collapse=" x ")))))}
 table(out)
 
 out <- rbind(out,(c(c(paste(els[1:(512)],collapse=" x "), paste(els[513:1024],collapse=" x ")))))
 
 o2x <- as.data.frame(out)
 XL <- rbind(XL,o2x)
 }
return(XL)
}



o2 <- getdata(fi[1])
for(z in fi[2:length(fi)]) {o2 <- rbind(o2,getdata(z))}





o2 <- na.omit(o2)
tb <- (table(paste(o2$V1,o2$V2,sep=";")))
NET <- data.frame(EDGE=tb,CROSS=names(tb))
 

NET$MUM <- gsub(".*;","",NET$CROSS)
NET$DAD <- gsub(";.*","",NET$CROSS)

NET$size <- lengths(regmatches(NET$CROSS, gregexpr(" x ", NET$CROSS)))
UZ <- unique(NET$size)
GEN=data.frame(size=UZ[order(UZ)],Generation=1:10)
NET <- merge(NET,GEN)

UN <- unique(NET$CROSS)
NAM <- data.frame(CROSS=UN,ID=1:length(UN))
NET <- (merge(NET,NAM))


NET$NODE <- paste("F",NET$Generation,"_",NET$ID,sep="")

NET2 <- NET
NET2$ CR <- gsub(";"," x ",NET2$CROSS)
SNAM <- NET2[,c("CR","NODE")]


for(x in 1:length(NET[,1]))
{
NET2$MUM[NET2$MUM == SNAM$CR[x]] <- SNAM$NODE[x]
NET2$DAD[NET2$DAD == SNAM$CR[x]] <- SNAM$NODE[x]
}


NET3 <- NET2[,4:9]



MUM <- NET3[,c("MUM","NODE","EDGE.Freq")]
DAD <- NET3[,c("DAD","NODE","EDGE.Freq")]

colnames(MUM)[1] <- "Child"
colnames(DAD)[1] <- "Child"

IG <- rbind(MUM,DAD)

IG$Cycle <- gsub("_.*","",IG$Child)
IG$LEVEL <- as.numeric(as.character(gsub("F","",IG$Cycle)))
IG$LEVEL[grep("F",IG$Cycle,invert=T)] <- 0

IS <- IG[IG$LEVEL ==0,]
IS$SCAL <- (IS$EDGE.Freq/max(IS$EDGE.Freq)*100)
IG$SCAL[IG$LEVEL==0] <- IS$SCAL

for(x in 1:max(IG$LEVEL))
{
IS <- IG[IG$LEVEL ==x,]
IS$SCAL <- (IS$EDGE.Freq/max(IS$EDGE.Freq)*100)
IG$SCAL[IG$LEVEL==x] <- IS$SCAL

}



UI <- unique(IG[IG$LEVEL ==0,])
UI <- aggregate(UI$EDGE.Freq,by=list(UI$Child),FUN=sum)
UI <- UI[order(UI$x,decreasing=T),]
UI$VAL <- 1:length(UI[,1])
UD <- data.frame(Child=UI[,1],VAL=UI$VAL)


for (x in 1:max(IG$LEVEL))
{
UI <- unique(IG[IG$LEVEL ==x,])
UI <- aggregate(UI$EDGE.Freq,by=list(UI$Child),FUN=sum)
UI <- UI[order(UI$x,decreasing=T),]
UI$VAL <- 1:length(UI[,1])
UX <- data.frame(Child=UI[,1],VAL=UI$VAL)
UD <- rbind(UD,UX)
}

IG <- merge(IG,UD)
##########################3

IG$Cycle2 <- gsub("_.*","",IG$NODE)
IG$LEVEL2 <- as.numeric(as.character(gsub("F","",IG$Cycle2)))
IG$LEVEL2[IG$LEVEL2 > 10] <- 0

UI <- unique(IG[IG$LEVEL2 ==1,])
UI <- aggregate(UI$EDGE.Freq,by=list(UI$NODE),FUN=sum)
UI <- UI[order(UI$x,decreasing=T),]
UI$VAL <- 1:length(UI[,1])
UD <- data.frame(NODE=UI[,1],VAL2=UI$VAL)

for (x in 2:max(IG$LEVEL2))
{
UI <- unique(IG[IG$LEVEL2 ==x,])
UI <- aggregate(UI$EDGE.Freq,by=list(UI$NODE),FUN=sum)
UI <- UI[order(UI$x,decreasing=T),]
UI$VAL <- 1:length(UI[,1])
UX <- data.frame(NODE=UI[,1],VAL2=UI$VAL)
UD <- rbind(UD,UX)
}

IG <- merge(IG,UD)

IS <- unique(IG[IG$LEVEL == 0,c(2,7)])

library(viridis)
library(ggplot2)


IG$LEVEL3 <- (((IG$VAL2-1) %/% 10)/5)+ IG$LEVEL2
IG$VAL3 <- ((IG$VAL2) %% 10)
IG$VALA <- ((IG$VAL) %% 10)
IG$VALA[IG$VALA == 0] <- 10
IG$VAL3[IG$VAL3 == 0] <- 10


IS <- unique(IG[IG$LEVEL == 0,c(2,13)])

CROSS <- ggplot(IG) +geom_segment(aes(x=LEVEL,y=VALA,xend=LEVEL3,yend=VAL3,alpha=SCAL/100),color="aquamarine4")+geom_point(aes(x=LEVEL3,y=VAL3,color=LEVEL2,size=SCAL))+theme_bw()+annotate("text",x=0,y=IS$VALA,label=IS$Child,hjust=1,size=3) +scale_color_viridis_c()+annotate("text",x=IG$LEVEL3,y=IG$VAL3-0.25,label=IG$EDGE.Freq,size=2)+ scale_x_continuous(breaks = seq(-2, 12)) +
  theme(axis.text.y=element_blank(),axis.ticks.y=element_blank()) +
  scale_y_continuous(breaks = NULL) +theme(legend.position="none")+ylab("")+xlab("Cycle")+xlim(-2,12)

fi <- list.files()

fi <- list.files()
fi <- fi[grep("Outfile",fi)]


d <- read.table(fi[1])
d$generation <- 0:10 
d$size <- strsplit(fi[1],",")[[1]][3]
d$split <- gsub(".txt","",strsplit(fi[1],",")[[1]][5])

data <- d

for(x in 2:length(fi))
{
d <- read.table(fi[x])
d$generation <- 0:10 
d$size <- strsplit(fi[x],",")[[1]][3]
d$split <- gsub(".txt","",strsplit(fi[x],",")[[1]][5])

data <- rbind(data,d)
}

colnames(data)[1:7] <- c("Index","Yield","Height","Heading","Zeleny","TKW","Cor")

pI <- ggplot(data)+geom_boxplot(aes(x=as.factor(generation),y=Index)) +theme_bw()+xlab("")
pY <- ggplot(data)+geom_boxplot(aes(x=as.factor(generation),y=Yield)) +theme_bw()+xlab("")
pS <- ggplot(data)+geom_boxplot(aes(x=as.factor(generation),y=Height)) +theme_bw()+xlab("")
pH <- ggplot(data)+geom_boxplot(aes(x=as.factor(generation),y=Heading)) +theme_bw()+xlab("")
pT <- ggplot(data)+geom_boxplot(aes(x=as.factor(generation),y=TKW)) +theme_bw()+xlab("")
pZ <- ggplot(data)+geom_boxplot(aes(x=as.factor(generation),y=Zeleny)) +theme_bw()+xlab("")
pC <- ggplot(data)+geom_boxplot(aes(x=as.factor(generation),y=Cor)) +theme_bw()+xlab("")

library("ggpubr")

#pdf("Shortindex.pdf",width=7,height=10)
ggarrange(CROSS, pI,pY,pS,pH,pT,pZ,pC, heights = c(2, 0.7,0.7,0.7,0.7,0.7,0.7,0.7),
          ncol = 1, nrow = 8)
#dev.off()



# select only crosses that contain 90% of the data
IG2 <- IG[IG$EDGE.Freq > quantile(IG$EDGE.Freq,0.99),]

# find how deep the cyles go
LEVS <- unique(IG2$Cycle2)

TOPLEV <- IG2[IG2$Cycle2 == LEVS[length(LEVS)],]
CROSSERS <- c(TOPLEV$NODE,TOPLEV$Child)

for( x in 1:(length(LEVS)-1))
{CROSSERS <- c(CROSSERS,IG2$Child[IG2$NODE %in% CROSSERS])}

CROSSERS <- unique(CROSSERS)


IG3 <- IG2[IG2$Child %in% CROSSERS,]

IGx <- IG3[IG3$LEVEL2 == max(IG3$LEVEL2),]
IGx <- rbind(IGx,IG3[IG3$NODE %in% IGx$Child,])
IGx <- rbind(IGx,IG3[IG3$NODE %in% IGx$Child,])
IGx <- rbind(IGx,IG3[IG3$NODE %in% IGx$Child,])
IGx <- unique(IGx)


GT <- ggplot(IGx) +geom_segment(aes(x=LEVEL,y=VAL,xend=LEVEL2,yend=VAL2,alpha=SCAL/100),color="aquamarine4")+geom_point(aes(x=LEVEL2,y=VAL2,color=LEVEL2,size=SCAL))+theme_bw()+annotate("text",x=0,y=IS$VALA,label=IS$Child,hjust=1) +scale_color_viridis_c()+annotate("text",x=IGx$LEVEL2,y=IGx$VAL2-0.25,label=IGx$EDGE.Freq)+ scale_x_continuous(breaks = seq(-2, 12)) +
  theme(axis.text.y=element_blank(),axis.ticks.y=element_blank()) +
  scale_y_continuous(breaks = NULL) +theme(legend.position="none")+ylab("")+xlab("Cycle")+xlim(-1,3)

pdf("Shortindex.pdf",width=7,height=10)
ggarrange(GT, pI,pY,pS,pH,pT,pZ,pC, heights = c(2, 0.7,0.7,0.7,0.7,0.7,0.7,0.7),
          ncol = 1, nrow = 8)
dev.off()
