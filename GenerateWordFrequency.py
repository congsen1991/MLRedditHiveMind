# -*- coding: utf-8 -*-
import sqlite3
import pandas as pd
from stop_words import get_stop_words
from sqlalchemy import create_engine
    
def getWordList(row):
    # take a row of a pandas dataframe as input
    # read sentence from current row, output words in a list.
    sentence = row['body']
    ans = ''
    
    for char in sentence:
        if char.isalpha():
            ans+=char
        else: ans+= ' '
    
    ans = ans.split()
    
    # using library stopwords to get stop words list
'''
"a", "about",  "above",  "after",  "again",  "against",  "all",  "am",  "an", 
"and",  "any",  "are",  "aren't",  "as",  "at",  "be",  "because", 
"been",  "before",  "being",  "below",  "between",  "both",  "but",  "by", 
"can't",  "cannot",  "could",  "couldn't",  "did",  "didn't",  "do",  "does", 
"doesn't",  "doing",  "don't",  "down",  "during",  "each",  "few",  "for", 
"from",  "further",  "had",  "hadn't",  "has",  "hasn't",  "have",  "haven't", 
"having",  "he",  "he'd",  "he'll",  "he's",  "her",  "here",  "here's", 
"hers",  "herself",  "him",  "himself",  "his",  "how",  "how's",  "i", 
"i'd",  "i'll",  "i'm",  "i've",  "if",  "in",  "into",  "is", 
"isn't",  "it",  "it's",  "its",  "itself",  "let's",  "me",  "more", 
"most",  "mustn't",  "my",  "myself",  "no",  "nor",  "not",  "of", 
"off",  "on",  "once",  "only",  "or",  "other",  "ought",  "our", 
"ours",  "ourselves",  "out",  "over",  "own",  "same",  "shan't",  "she", 
"she'd",  "she'll",  "she's",  "should",  "shouldn't",  "so",  "some",  "such", 
"than",  "that",  "that's",  "the",  "their",  "theirs",  "them",  "themselves", 
"then",  "there",  "there's",  "these",  "they",  "they'd",  "they'll",  "they're", 
"they've",  "this",  "those",  "through",  "to",  "too",  "under",  "until", 
"up",  "very",  "was",  "wasn't",  "we",  "we'd",  "we'll",  "we're", 
"we've",  "were",  "weren't",  "what",  "what's",  "when",  "when's",  "where", 
"where's",  "which",  "while",  "who",  "who's",  "whom",  "why",  "why's", 
"with",  "won't",  "would",  "wouldn't",  "you",  "you'd",  "you'll",  "you're", 
"you've",  "your",  "yours",  "yourself",  "yourselves", 
'''
    stop_words = set(get_stop_words('english'))
    
    return [word for word in ans if word not in stop_words and len(word)>2]
    

def Solution(subredditName, tableName):
    # a function take input as subreddit name we are interested
    # query from subreddit database, fetch body and score
    # export the data into a new pandas dataframe
    
    sql_conn = sqlite3.connect('database/'+subredditName+'.sqlite')

    query = "SELECT body,score FROM " + tableName
    Users = pd.read_sql(query, sql_conn)

    return pd.DataFrame(Users)

def makeDict(row):
    # a simple function, gather all words from current row, 
    # add it to a hashmap
    curWordList = row['wordList']
    for word in curWordList:
        curDict[word] = curDict.get(word,0) + 1
    
subredditOfInterest = ['worldnews',
                       'technology',
                       'politics']

# tablenames are defined to be consistent with previous script
tableNames = [subreddit+'1' for subreddit in subredditOfInterest]

worldnews,technology,politics=0,1,2
# dataList is for 3 pandas dataframe, each one is for a subreddit
# the order is the same as above 
dataList = []

for index in range(len(subredditOfInterest)):
    # read data from sqlite database of each subreddit, 
    # put them in pandas dataframe
    subredditName = subredditOfInterest[index]
    tableName = tableNames[index]
    curDF = Solution(subredditName, tableName)
    dataList.append(curDF)
    
    # Add another column to pandas dataframe, the list of words
    curDF['wordList'] = curDF.apply(getWordList,axis=1)
    
    # make a local hashmap for current subreddit    
    curDict = {}
    
    # apply function makeDict to all wordList of current subreddit
    # so that the words are processed and count stored in curDict
    curDF.apply(makeDict, axis=1)
    
    # make a new pandas dataframe called countDF
    # which is equivalent to the hashmap
    countDF = pd.DataFrame({'count':curDict.values(), 'key':curDict.keys()})
    
    # Sort countDF according to counts
    countDF = countDF.sort_values(by='count',ascending = False)
    
    # write them to csv
    countDF.to_csv('data/'+subredditName+'.csv', sep=',', encoding='utf-8')
    countDF[0:100].to_csv('data/'+subredditName+'_100.csv', sep=',', encoding='utf-8')
    


    




