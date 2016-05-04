import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os, pickle
import glob
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import train_test_split
from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from sklearn.tree import export_graphviz
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectFromModel
def convertScore(row,minVal,maxVal):
    score = row['score']
    if score<0: return 1
    if score<10: return 2
    if score<100: return 3
    if score<1000: return 4
    return 5
def readData(subject):
    df = pd.read_pickle(subject + '.pickle')
    #df.to_csv('../data/' + subject + '.csv', sep=',', encoding='utf-8' )
for file in glob.glob("processedData/*Data.pickle"):
    subject=("../processData/" + file)[15:-7]

    print subject
    # cardataframe = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data',header=None)
    #americaDataFrame = decisionTree("americaData.pickle");
    americaDataFrame = pd.read_pickle(subject+ '.pickle')
    maxVal = americaDataFrame['score'].max()
    minVal = americaDataFrame['score'].min()
    print maxVal,minVal
    americaDataFrame['newScore']= americaDataFrame.apply(convertScore,args=[minVal,maxVal],axis=1)
    print americaDataFrame.head()
    americaDataFrame.columns = ['body', 'score', 'overallpol','stdPol', 'overallSub', 'stdSub', 'polRange', 'subRange', 'wordCount', 'bigWords', 'sentLen', 'targetPol', 'targetSub','newScore']
    Y = americaDataFrame['newScore'].values
    X = americaDataFrame[['overallpol','stdPol', 'overallSub', 'stdSub', 'polRange', 'subRange', 'wordCount', 'bigWords', 'sentLen', 'targetPol', 'targetSub']].values
    print Y

    # xSet = [2,3,4,5,6,7,8,9,10,11,12]
    # y=1
    # X = americaDataFrame.ix[:,xSet]
    # Y = americaDataFrame.ix[:,y]
    X = np.array(X).astype(float)
    Y = np.array(Y).astype(float)
    print X, Y
    #sc = StandardScaler()
    ##sc=sc.fit(X)
    #X = sc.transform(X)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
    # sc = StandardScaler()
    # sc=sc.fit(X_train)
    # X_train = sc.transform(X_train)
    # X_test = sc.transform(X_test)

    tree = DecisionTreeClassifier(criterion = 'entropy', max_depth=3, random_state=0)
    selector = RFE(tree, 2, step=1)
    selector = selector.fit(X_train, Y_train)
    selector.support_
    selector.ranking_
    tree.fit(X_train, Y_train)
    export_graphviz(tree, out_file= subject + '.dot',feature_names=['overallpol','stdPol', 'overallSub', 'stdSub', 'polRange', 'subRange', 'wordCount', 'bigWords', 'sentLen', 'targetPol', 'targetSub'])
           #  self.X=self.data.ix[:,xSet]
    #         self.Y=self.data.ix[:,y]


