######################################
#Read data from pickle
######################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
# encode the result
def encodeData(input, prediction):
    for i in range(len(input)):
        if input[i] >= 5 :
            input[i] = 3
        elif input[i] <= -5:
            input[i] = 0
        elif input[i] < 5 and input[i] >= 0:
            input[i] = 2
        else:
            input[i] = 1
    for i in range(len(prediction)):
        if prediction[i] >= 5 :
            prediction[i] = 3
        elif prediction[i] <= -5:
            prediction[i] = 0
        elif prediction[i] < 5 and input[i] >= 0:
            prediction[i] = 2
        else:
            prediction[i] = 1

#read data
americaDataFrame = pd.read_pickle("americaData.pickle")
americaDataFrame.columns = ['body', 'score', 'overallpol','stdPol', 'overallSub', 'stdSub', 'polRange', 'subRange', 'wordCount', 'bigWords', 'sentLen', 'targetPol', 'targetSub']
y = americaDataFrame['score'].values
x = americaDataFrame[['overallpol','stdPol', 'overallSub', 'stdSub', 'polRange', 'subRange', 'wordCount', 'bigWords', 'sentLen', 'targetPol', 'targetSub']].values

#scale data
from sklearn import preprocessing
import numpy as np
y_np = np.array(y)
x_np = np.array(x)

X = preprocessing.scale(x_np)
Y = preprocessing.scale(y_np)

# print X
# print Y
# min_max_scaler = preprocessing.MinMaxScaler()

# X = min_max_scaler.fit_transform(x_np)
# Y = min_max_scaler.fit_transform(y_np)


from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split
######################################
#setup the dataset (supervised classification training) for neural network
######################################
from pybrain.utilities import percentError
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules import SoftmaxLayer
from pybrain.datasets.classification import ClassificationDataSet
from pybrain.datasets import SupervisedDataSet


ds = SupervisedDataSet(11, 1)
for i in range(len(X)):
    ds.addSample(X[i], Y[i])
# #split the dataset
trainData, testData = ds.splitWithProportion(0.70)


# ###################################
# #Creating a Neural Network
# ###################################
# # build nerual net with 21 inputs, 5 hidden neuron and 1 output neuron
net = buildNetwork(11,7,1,bias=True)
trainer = BackpropTrainer(net, trainData)
trnerr, valerr = trainer.trainUntilConvergence(dataset = trainData, maxEpochs = 50)

# #evaluate the error rate on training data
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
train_out = net.activateOnDataset(trainData) #return the output


#encodeData(train_out, trainData['target'])

#calculate the error
train_error = percentError( train_out, trainData['target'])
train_mse = mean_squared_error(trainData['target'], train_out)

#testdata
test_out = net.activateOnDataset(testData)
#encodeData(test_out, testData['target'])
test_error = percentError( test_out, testData['target'])
test_mse = mean_squared_error(testData['target'], test_out)

# print('neural network training accuracies %.2f'
#      % (train_acc))
# print('neural network test accuracies %.2f'
#      % (test_acc))

print('neural network traindata MSE %.2f' % (train_mse))
print('neural network testdata MSE %.2f' % (test_mse))









