######################################
#Read data from pickle
######################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
# cardataframe = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data',header=None)
americaDataFrame = pd.read_pickle("americaData.pickle")

americaDataFrame.columns = ['body', 'score', 'overallpol','stdPol', 'overallSub', 'stdSub', 'polRange', 'subRange', 'wordCount', 'bigWords', 'sentLen', 'targetPol', 'targetSub']
Y = americaDataFrame['score'].values
X = americaDataFrame[['overallpol','stdPol', 'overallSub', 'stdSub', 'polRange', 'subRange', 'wordCount', 'bigWords', 'sentLen', 'targetPol', 'targetSub']].values

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
trainData, testData = ds.splitWithProportion(0.60)


# ###################################
# #Creating a Neural Network
# ###################################
# # build nerual net with 21 inputs, 5 hidden neuron and 6 output neuron
net = buildNetwork(11,5,1,bias=True)
trainer = BackpropTrainer(net, trainData)
trnerr, valerr = trainer.trainUntilConvergence(dataset = trainData, maxEpochs = 50)

# #evaluate the error rate on training data
from sklearn.metrics import accuracy_score
train_out = net.activateOnDataset(trainData) #return the output

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

encodeData(train_out, trainData['target'])

#calculate the error
train_error = percentError( train_out, trainData['target'])
train_acc = accuracy_score( train_out, trainData['target'])

#testdata
test_out = net.activateOnDataset(testData)
encodeData(test_out, testData['target'])
test_error = percentError( test_out, testData['target'])
test_acc = accuracy_score( test_out, testData['target'])

print('neural network training accuracies %.2f'
     % (train_acc))
print('neural network test accuracies %.2f'
     % (test_acc))








