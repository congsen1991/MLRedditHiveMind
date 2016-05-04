if (!require("wordcloud")) {
  install.packages("wordcloud")
  library(wordcloud)
}
if (!require("RColorBrewer")) {
  install.packages("RColorBrewer")
  library(RColorBrewer)
}

setwd("/Volumes/Transcend/MLRedditHiveMind/")

fileNames <- c('politics_100','technology_100','worldnews_100')

wordsPlot <- function(fileName){
df <- read.csv(paste("data/",fileName,".csv",sep=''),header = TRUE)
df <- df[,-1]

png(paste("imgs/",fileName,".png",sep=""))
wordcloud(toupper(df$key),
          df$count,
          scale=c(5,.1),
          random.order=F,
          rot.per=.10,
          max.words=5000,
          colors=brewer.pal(8, "Dark2"),
          family="Avenir Next Condensed Bold",
          random.color=T)
dev.off()
}

for (i in 1:3){
  wordsPlot(fileNames[i])
}


fileNames <- c('politics','technology','worldnews')

histPlot <- function(fileName){
  df <- read.csv(paste("data/",fileName,".csv",sep=''),header = TRUE)
  df <- df[,-1]
  print(summary(df))
  png(paste("imgs/",fileName,"_count_hist.png",sep=""))
  hist(df$count)
  dev.off()
  
  countdf <- read.csv(paste("data/",fileName,"_raw.csv",sep=''),header = FALSE)
  png(paste("imgs/",fileName,"_raw_count_hist.png",sep=""))
  hist(df[,1])
  dev.off()  
  
}

for (i in 1:3){
  histPlot(fileNames[i])
}

