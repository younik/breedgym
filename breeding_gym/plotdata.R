
i <- list.files()
fi <- fi[grep("50",fi)]
d <- read.table(fi[1])
colnames(d) <- c("INDEX","Yield","Height","Heading","TKW","Zeleny","Similarity")
d$generation <- 1:21
d$parents <- substr(fi[1],10,11)

data <- d

for (x in 2:length(fi))
{
d <- read.table(fi[x])
colnames(d) <- c("INDEX","Yield","Height","Heading","TKW","Zeleny","Similarity")
d$generation <- 1:21
d$parents <- substr(fi[x],10,11)
data <- rbind(data,d)
}


ggplot(data) +geom_boxplot(aes(x=sprintf("%02d", data$generation),y=Yield,fill=parents))
