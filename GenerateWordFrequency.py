# -*- coding: utf-8 -*-
import sqlite3
import pandas as pd
from stop_words import get_stop_words
from sqlalchemy import create_engine
    
def getWordList(row):
    # take any sentence as input, output words in a list.
    sentence = row['body']
    ans = ''
    
    for char in sentence:
        if char.isalpha():
            ans+=char
        else: ans+= ' '
    
    ans = ans.split()
    
    stop_words = set(get_stop_words('english'))
    
    return [word for word in ans if word not in stop_words and len(word)>2]
    

def Solution(subredditName, tableName):
    # a function take input as subreddit name we are interested
    # query from original database, fetch body and score
    # export the data into a new sqlite database
    # the name of new database is the same as subreddit name
    # the name of table is subredditname+1
    
    sql_conn = sqlite3.connect(subredditName+'.sqlite')

    query = "SELECT body,score FROM " + tableName
    Users = pd.read_sql(query, sql_conn)

    return pd.DataFrame(Users)

def makeDict(row):
    curWordList = row['wordList']
    for word in curWordList:
        curDict[word] = curDict.get(word,0) + 1
    
subredditOfInterest = ['worldnews',
                       'technology',
                       'politics']
                       
tableNames = [subreddit+'1' for subreddit in subredditOfInterest]

worldnews,technology,politics=0,1,2
dataList = []

for index in range(len(subredditOfInterest)):
    subredditName = subredditOfInterest[index]
    tableName = tableNames[index]
    curDF = Solution(subredditName, tableName)
    dataList.append(curDF)
    
    curDF['wordList'] = curDF.apply(getWordList,axis=1)
    curDict = {}
    curDF.apply(makeDict, axis=1)
    countDF = pd.DataFrame({'count':curDict.values(), 'key':curDict.keys()})
    countDF = countDF.sort_values(by='count',ascending = False)
    countDF.to_csv('data/'+subredditName+'.csv', sep=',', encoding='utf-8')
    countDF[0:100].to_csv('data/'+subredditName+'_100.csv', sep=',', encoding='utf-8')
    


    




