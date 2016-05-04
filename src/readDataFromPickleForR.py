# -*- coding: utf-8 -*-
"""
Created on Wed May  4 11:24:53 2016

@author: chao
"""

######################################
#Read data from pickle
######################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os,glob
# cardataframe = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data',header=None)

def readData(subject):
    df = pd.read_pickle(subject+'.pickle')
    df.to_csv('../data/'+subject+'.csv', sep=',', encoding='utf-8')
    
for file in glob.glob("../processedData/*Data.pickle"):
    subject= ("../processedData/"+file)[17:-7]
    readData(subject)