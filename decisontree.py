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

# class decisionTree(object):
#     def __init__(self, fileName):
#         #Pull in data
#         self.data=pd.read_pickle(fileName)
#         self.data=pd.DataFrame(self.data)
# 
#     
#     def impute(self, col):
#         #Impute medians to address NaN
#         imput=preprocessing.Imputer(missing_values='NaN',strategy='median',axis = 1)
#         tmp=imput.fit_transform(self.data[col]).T
#         self.data[col]=tmp
# 	
#     def printData(self):
#         print self.data.head()
# 
# 
#     def createTestSet(self,xSet,y):
# 	    #Create trainning and testing sets
#         self.X=self.data.ix[:,xSet]
#         self.Y=self.data.ix[:,y]
#         self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.X, self.Y, test_size=0.3, random_state=0)
# 
# 
#     def scale(self):
#         """ The data is scaled in preparation for creating the DT.  WHY ?  
#         WHAT IS THE DIFFERENCE BETWEEN SCALING AND NORMALIZATION?"""
#         sc = StandardScaler()
#         sc=sc.fit(X_train)
#         self.X_train = sc.transform(self.X_train)
#         self.X_test = sc.transform(self.X_test)
# 
#     def buildTree(self,depth):
#         #Here, we define the parameters of our tree and use a feature selection algorithm (RFE) to pick out the strongest features.
# 
#         self.tree = DecisionTreeClassifier(criterion = 'entropy', max_depth=depth, random_state=0)
#         selector = RFE(self.tree, 2, step=1)
#         selector = selector.fit(self.X_train, self.Y_train)
#         selector.support_
#         selector.ranking_
# 
#     def fitTree(self):
#         # We then fit the tree to our training data 
# 
#         self.tree.fit(self.X_train, self.Y_train)
# 
#     def visTree(self):
#         # Now we visualize our tree
# 
#         export_graphviz(self.tree, out_file='tree.dot',feature_names=['pclass', 'age','sibsp','parch','fare','male','female'])
# 
#     def pred(self):
#         # Let's make a prediction
# 
#         self.Y_pred=self.tree.predict(self.X_test)
# 
#     def confusionMatrix(self):
#         # Now we calculate our accuracy and create a confusion matrix of our results
# 
#         print('Accuracy: %.2f' % accuracy_score(self.Y_test,self.Y_pred))
#         
#         confmat=confusion_matrix(y_true=self.Y_test, y_pred=self.Y_pred)
#         print(confmat)
######################################
#Read data from pickle
######################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

def convertScore(row,minVal,maxVal):
    score = row['score']
    if score<0: return 1
    if score<10: return 2
    if score<100: return 3
    if score<1000: return 4
    return 5
    
# cardataframe = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data',header=None)
#americaDataFrame = decisionTree("americaData.pickle");
americaDataFrame = pd.read_pickle("americaData.pickle")
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
export_graphviz(tree, out_file='tree.dot',feature_names=['overallpol','stdPol', 'overallSub', 'stdSub', 'polRange', 'subRange', 'wordCount', 'bigWords', 'sentLen', 'targetPol', 'targetSub'])
       #  self.X=self.data.ix[:,xSet]
#         self.Y=self.data.ix[:,y]


