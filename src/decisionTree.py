import numpy as np
import pandas as pd
import glob
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

class decisionTree(object):
    def __init__(self, subject, header = 0):
        #Pull in data

        df = pd.read_pickle(subject+'.pickle')
        self.data=pd.DataFrame(df)
        self.data.dropna()

    def writeToCsv(self, fileName):
        self.data.to_csv(fileName,sep=',')

    def replace(self,col,tobe,be):
        self.data[col]=self.data[col].replace(to_replace=tobe,value=be)

    def dupCol(self,oldCol,newCol):
        self.data[newCol] = self.data[oldCol]
    
    def impute(self, col):
        #Impute medians to address NaN
        imput=preprocessing.Imputer(missing_values='NaN',strategy='median',axis = 1)
        tmp=imput.fit_transform(self.data[col]).T
        self.data[col]=tmp
	
    def printData(self):
        print self.data.head()

    def dropCol(self,col):
        #Drop first column
        self.data=self.data.drop(col,axis=1)

    def createTestSet(self,xSet,y):
	    #Create trainning and testing sets
        self.X=self.data.ix[:,xSet]
        self.Y=self.data.ix[:,y]
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.X, self.Y, test_size=0.3, random_state=0)


    def scale(self):
        """ The data is scaled in preparation for creating the DT.  WHY ?  
        WHAT IS THE DIFFERENCE BETWEEN SCALING AND NORMALIZATION?"""
        sc = StandardScaler()
        sc=sc.fit(X_train)
        self.X_train = sc.transform(self.X_train)
        self.X_test = sc.transform(self.X_test)

    def buildTree(self,depth):
        #Here, we define the parameters of our tree and use a feature selection algorithm (RFE) to pick out the strongest features.

        self.tree = DecisionTreeClassifier(criterion = 'entropy', max_depth=depth, random_state=0)
        selector = RFE(self.tree, 2, step=1)
        selector = selector.fit(self.X_train, self.Y_train)
        selector.support_
        selector.ranking_

    def fitTree(self):
        # We then fit the tree to our training data 

        self.tree.fit(self.X_train, self.Y_train)

    def visTree(self,fileName,xSet):
        # Now we visualize our tree

        export_graphviz(self.tree, out_file=fileName,feature_names=[self.colNameMap[x] for x in xSet])

    def pred(self):
        # Let's make a prediction

        self.Y_pred=self.tree.predict(self.X_test)

    def confusionMatrix(self):
        # Now we calculate our accuracy and create a confusion matrix of our results

        print('Accuracy: %.2f' % accuracy_score(self.Y_test,self.Y_pred))
        
        confmat=confusion_matrix(y_true=self.Y_test, y_pred=self.Y_pred)
        print(confmat)

    def getColNames(self):
        self.colNames = self.data.dtypes.index
        self.colNameMap = {}
        for i in range(len(self.colNames)):
            self.colNameMap[i]=self.colNames[i]



##################################################
#  feature selection
##################################################

  
for file in glob.glob("../processedData/*Data.pickle"):
    subject= ("../processedData/"+file)[17:-7]

d = decisionTree(subject)
d.printData()

#d.writeToCsv('imputed_data.csv')
d.getColNames()
print d.colNames
d.printData()

xSet = [2, 3,4,5,6,7,8,9,10,11,12]
y = 1
d.createTestSet(xSet,y)

d.buildTree(3)
d.fitTree()
d.visTree("no_gender.dot",xSet)

d.pred()
d.confusionMatrix()

