if (!require("wordcloud")) {
  install.packages("wordcloud")
  library(wordcloud)
}
if (!require("RColorBrewer")) {
  install.packages("RColorBrewer")
  library(RColorBrewer)
}


df <- data.frame(word = c('a','b','c','d'),
                 wordCount = c(2,5,3,6))

wordcloud(toupper(df$word),
          df$wordCount,
          scale=c(5,.1),
          random.order=F,
          rot.per=.10,
          max.words=5000,
          colors=brewer.pal(8, "Dark2"),
          family="Avenir Next Condensed Bold",
          random.color=T)