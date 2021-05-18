##### Elon Musk Tweet #####

library(tm)
library(SnowballC)
library(wordcloud)
library(RWeka)	
library(qdap)		
library(textir)
library(maptpx)
library(data.table)
library(stringr)
library(slam)
library(ggplot2)

tweet <- read.csv(file.choose())
str(tweet)
#tweet[] <- lapply(tweet, as.character)
text <- tweet$Text
head(text,10)

text <- tolower(text)

# Remove mentions, urls, emojis, numbers, punctuations, etc.
text <- gsub("@\\w+", "", text)
text <- gsub("https?://.+", "", text)
text <- gsub("\\d+\\w*\\d*", "", text)
text <- gsub("#\\w+", "", text)
text <- gsub("[^\x01-\x7F]", "", text)
text <- gsub("[[:punct:]]", " ", text)

# Remove spaces and newlines
text <- gsub("\n", " ", text)
text <- gsub("^\\s+", "", text)
text <- gsub("\\s+$", "", text)
text <- gsub("[ |\t]+", " ", text)

corpus <- Corpus(VectorSource(text))
inspect(corpus[c(1:10)])

# Loading positive, negatice, stoprwords

pos.words=scan(file.choose(), what="character", comment.char=";")
neg.words=scan(file.choose(), what="character", comment.char=";")
stopwdrds = readLines(file.choose())

# Cleaning Data

corpus = tm_map(corpus, tolower)	
corpus = tm_map(corpus, removePunctuation)
#my_punctuation <- c(".","'")
corpus = tm_map(corpus, removeNumbers)
corpus = tm_map(corpus, removeWords, c(stopwords("english"),stopwdrds))
#toSpace <- content_transformer(function (x , pattern ) gsub(pattern, " ", x))
#corpus <- tm_map(corpus,toSpace,"...")
corpus = tm_map(corpus, stripWhitespace) 	# removes white space
corpus = tm_map(corpus, stemDocument)

# Term Document Matrix

tdm0 <- TermDocumentMatrix(corpus)
inspect(tdm0)

tweet_dtm <- as.matrix(tdm0)
tweet_v <- sort(rowSums(tweet_dtm),decreasing = TRUE)
tweet_d <- data.frame(word = names(tweet_v),freq=tweet_v)
head(tweet_d,5)

# Term document matrix with inverse frequency

tdm1 <- TermDocumentMatrix(corpus,control = list(weighting = function(p) weightTfIdf(p,normalize = T)))
inspect(tdm1)

# Removing empty documents

a0 <- NULL
a1 <- NULL

# getting the indexes of documents having count of words = 0
for (i1 in 1:ncol(tdm0))
{ if (sum(tdm0[, i1]) == 0) {a0 = c(a0, i1)} }
for (i1 in 1:ncol(tdm1))
{ if (sum(tdm1[, i1]) == 0) {a1 = c(a1, i1)} }

# Removing empty docs 
tdm0 <- tdm0[,-a0]
tdm1 <- tdm1[,-a1]

# Document term matrix 
dtm0 <- t(tdm0)
dtm1 <- t(tdm1)

# Defining functions for words clouds


makewordc = function(x){	
  freq = sort(rowSums(as.matrix(x)),decreasing = TRUE)
  freq.df = data.frame(word=names(freq), freq=freq)
  windows()
  wordcloud(freq.df$word[1:120], freq.df$freq[1:120],scale = c(4,.5),random.order = F, colors=1:10)
} 

# Making positive wordcloud function 
makeposwordc = function(x){
  freq = sort(rowSums(as.matrix(x)),decreasing = TRUE)
  # matching positive words
  pos.matches = match(names(freq), c(pos.words,"approvals"))
  pos.matches = !is.na(pos.matches)
  freq_pos <- freq[pos.matches]
  names <- names(freq_pos)
  windows()
  wordcloud(names,freq_pos,scale=c(4,.5),colors = brewer.pal(8,"Dark2"))
}

# Making negatice wordcloud function
makenegwordc = function(x){	
  freq = sort(rowSums(as.matrix(x)),decreasing = TRUE)
  # matching positive words
  neg.matches = match(names(freq), neg.words)
  neg.matches = !is.na(neg.matches)
  freq_neg <- freq[neg.matches]
  names <- names(freq_neg)
  windows()
  wordcloud(names[1:120],freq_neg[1:120],scale=c(4,.5),colors = brewer.pal(8,"Dark2"))
}



words_bar_plot <- function(x){
  freq = sort(rowSums(as.matrix(x)),decreasing = TRUE)
  freq.df = data.frame(word=names(freq), freq=freq)
  head(freq.df, 20)
  library(ggplot2)
  windows()
  ggplot(head(freq.df,50), aes(reorder(word,freq), freq)) +
    geom_bar(stat = "identity") + coord_flip() +
    xlab("Words") + ylab("Frequency") +
    ggtitle("Most frequent words")
  
}

pos_words_bar_plot <- function(x){
  pos.matches = match(colnames(x), pos.words)
  pos.matches = !is.na(pos.matches)
  pos_words_freq = as.data.frame(apply(x, 2, sum)[pos.matches])
  colnames(pos_words_freq)<-"freq"
  pos_words_freq["word"] <- rownames(pos_words_freq)
  # Sorting the words in deceasing order of their frequency
  pos_words_freq <- pos_words_freq[order(pos_words_freq$freq,decreasing=T),]
  windows()
  ggplot(head(pos_words_freq,30), aes(reorder(word,freq), freq)) +
    geom_bar(stat = "identity") + coord_flip() +
    xlab("Positive words") + ylab("Frequency") +
    ggtitle("Most frequent positive words")
}
neg_words_bar_plot <- function(x){
  neg.matches = match(colnames(x), neg.words)
  neg.matches = !is.na(neg.matches)
  neg_words_freq = as.data.frame(apply(x, 2, sum)[neg.matches])
  colnames(neg_words_freq)<-"freq"
  neg_words_freq["word"] <- rownames(neg_words_freq)
  # Sorting the words in deceasing order of their frequency
  neg_words_freq <- neg_words_freq[order(neg_words_freq$freq,decreasing=T),]
  windows()
  ggplot(head(neg_words_freq,30), aes(reorder(word,freq), freq)) +
    geom_bar(stat = "identity") + coord_flip() +
    xlab("words") + ylab("Frequency") +
    ggtitle("Most frequent negative words")
}


# Word Cloud
makewordc(tdm0)
title(sub = "UNIGRAM - Wordcloud using TF")

# Frequency Bar plot
words_bar_plot(tdm0)

# Word cloud - TFIDF - Unigram
makewordc(tdm1)
title(sub = "UNIGRAM - Wordcloud using TFIDF")

# Frequency Barplot - TFIDF - Unigram
words_bar_plot(tdm1)

# Sentiment Analysis
install.packages("syuzhet")
library("syuzhet")
#tweet[] <- lapply(tweet,as.character)
tweet_sa <- get_sentences(text)

tweet_syuzhet <- get_sentiment(tweet_sa,method = "syuzhet")
head(tweet_syuzhet)
summary(tweet_syuzhet)

tweet_bing <- get_sentiment(tweet_sa,method = "bing")
head(tweet_bing)
summary(tweet_bing)

tweet_afinn <- get_sentiment(tweet_sa,method = "afinn")
head(tweet_afinn)
summary(tweet_afinn)

# comparison between 3 methods
rbind(
  sign(head(tweet_syuzhet)),
  sign(head(tweet_bing)),
  sign(head(tweet_afinn))
)

##### Prodcut Review and Emotion Mining #####

library(rvest)
library(XML)
library(magrittr)
library(rapportools)

aurl <- "https://www.amazon.in/Amazfit-Bip-Smart-Watch-HD-Color-Display-Sports-Modes-Breathing/product-reviews/B08HHCDN97/ref=cm_cr_arp_d_paging_btm_next_2?ie=UTF8&reviewerType=all_reviews&pageNumber"
amazon_reviews <- NULL
for (i in 1:20){
  murl <- read_html(as.character(paste(aurl,i,sep="=")))
  rev <- murl %>%
    html_nodes(".review-text") %>%
    html_text()
  amazon_reviews <- c(amazon_reviews,rev)
}
write.table(amazon_reviews,"bipu.txt",row.names = F)

##### Inporting file

reviews <- readLines(file.choose())
head(reviews,10)

#reviews <- stemDocument(reviews)
vc1 <- get_sentences(reviews)
head(vc1,50)


##### removing empty lines

empty_lines = grepl('^\\s*$',vc1)
vc1 = vc1[! empty_lines]
vc1 = paste(vc1,collapse = '\n')

ac <- get_sentences(vc1)

##### Emotion Mining

d <- get_nrc_sentiment(ac)
head(d,10)

barplot(colSums(d),las = 2, col = rainbow(10), ylab = 'count',main = "Emotion Score")

sentiment_vector <- get_sentiment(ac,method = "afinn")
plot(sentiment_vector, type = "l", main = "Plot Trajectory", xlab = "Review", ylab = "Emotional Valence")
abline(h=0, col = "red")

negative <- ac[which.min(sentiment_vector)]
negative

positive <- ac[which.max(sentiment_vector)]
positive

