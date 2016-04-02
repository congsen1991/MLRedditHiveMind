if (!require("wordcloud")) {
  install.packages("wordcloud")
  library(wordcloud)
}
if (!require("RColorBrewer")) {
  install.packages("RColorBrewer")
  library(RColorBrewer)
}

setwd("/Volumes/Transcend/MLRedditHiveMind/")

df <- read.csv("politics_100.csv",header = TRUE)
df <- df[,-1]

wordcloud(toupper(df$key),
          df$count,
          scale=c(5,.1),
          random.order=F,
          rot.per=.10,
          max.words=5000,
          colors=brewer.pal(8, "Dark2"),
          family="Avenir Next Condensed Bold",
          random.color=T)