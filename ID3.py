#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 10 09:55:48 2021

@author: cameronleong
"""

import decision_tree_class as t
import numpy as np
import random as r  
import pandas as pd
import sys
raw_data1 = pd.read_csv("train.csv")
raw_data1 = pd.DataFrame(raw_data1).to_numpy()
raw_data2 = pd.read_csv("heart.csv")
raw_data2 = pd.DataFrame(raw_data2).to_numpy()


#categories for titanic dataset
#[  (0)passenger class1, (1)passenger class 2 (2)Passenger class 3  (3)male,  (4)child,
#  (5)adult,  (6)elderly,
# (7)>0 siblings/spouses,  (8) >1 parent/child,  (9)low fare, (10)Cherbourg 
# (11)Queenstown (12)Southampton ]

male = raw_data1[0][4] 
def binarize_titanic(x):
    y = np.zeros((len(x),15))   #reorganized data stored in y
    for i in range(len(x)):
        #target values
        y[i][14] = x[i][1]
        y[i][13] = x[i][0]  #passenger number
        #passenger class
        if x[i][2] == 1:
            y[i][0] = 1
        elif x[i][2] == 2:
            y[i][1] = 1
        else:
            y[i][2] = 1
        #sex
        if x[i][4] == male:
            y[i][3] = 1
        #age
        if x[i][5]<13:
            y[i][4] = 1
        elif x[i][5] >= 65:
            y[i][6] = 1
        else:
            y[i][5] = 1
        #siblings/spouses
        if x[i][6] > 0:
            y[i][7] = 1
        #parents/children
        if x[i][7] > 1:
            y[i][8] = 1
        #low fare (avg is just above 32)
        if x[i][9] <= 32:
            y[i][9] = 1
        #city
        if x[i][11] == 'C':
            y[i][10] = 1
        elif x[i][11] == 'Q':
            y[i][11] = 1
        else:
            y[i][12] = 1
    return y

#categories for heart attack dataset
#[(0) age > 54,  (1) sex, (2) chest pain type 0,  (3) chest pain type 1
#  (4) chest pain type 2,  (5) chest pain type 3,  (6) resting blood pressure > 132,
#  (7) cholesterol > 246 (8) fasting blood sugar > 120,  
#  (9) resting electrocardiographic results  (10) max heart rate > 150,
#  (11) exercise induced angina,  (12) old peak > 1 (13) slope = 0,  
#(14) slope = 1,  (15) slope = 2, (16) num of major vessels > 1
#  (17) Thal rate > 0, (18) Target Value]
def binarize_hearts(x):
    z = np.zeros((len(x),19))
    #target values
    for i in range(len(x)):
        z[i][18] = x[i][13]
        #age > 54
        if x[i][0] > 54:
            z[i][0] = 1
        #sex
        if x[i][1] == 1:
            z[i][1] = 1
        #chest pain types
        if x[i][2] == 0:
            z[i][2] = 1
        elif x[i][2] == 1:
            z[i][3] = 1
        elif x[i][2] == 2:
            z[i][4] = 1
        elif x[i][2] == 3:
            z[i][5] = 1
        #resting blood pressure > 132
        if x[i][3] > 132:
            z[i][6] = 1
        #cholesterol > 246
        if x[i][4] > 246:
            z[i][7] = 1
        #fasting blood sugar > 120
        if x[i][5] == 1:
            z[i][8] = 1
        #resting ecg results
        if x[i][6] == 1:
            z[i][9] = 1
        #max heart rate > 150
        if x[i][7] > 140:
            z[i][10] = 1
        #exercise induced angina
        if x[i][8] == 1:
            z[i][11] = 1
        #old peak > 1
        if x[i][9] > 1:
            z[i][12] = 1
        #slopes
        if x[i][10] == 0:
            z[i][13] = 1
        elif x[i][10] == 1:
            z[i][14] = 1
        elif x[i][10] == 2:
            z[i][15] = 1
        #number of major blood vessels > 1
        if x[i][11] > 1:
            z[i][16] = 1
        if x[i][12] > 0:
            z[i][17] = 1
    return z
          
def initialize_data(raw_data,binarize,targets_idx):
    data = binarize(raw_data)       #convert data into binary categories
    test_size = round(.3*len(data))         #use 30% of data to test
    train_size = round(.7*len(data))     #use 70% of the data to train
    train = r.choices(data,k = train_size)  
    test = r.choices(data, k = test_size)
    train_targets = []
    test_targets = []
    for i in range(train_size):
        train_targets.append(train[i][targets_idx])  #target values in column 13
    for i in range(test_size):
        test_targets.append(test[i][targets_idx])
    train = np.delete(train,(targets_idx - 1),axis = 1)
    test = np.delete(test,(targets_idx - 1),axis = 1)
    return train,train_targets,test,test_targets


#organize data for each dataset into test, test targets, train, and train targets
train1,train_targets1,test1,test_targets1 = initialize_data(raw_data1,binarize_titanic,14)
train2,train_targets2,test2,test_targets2 = initialize_data(raw_data2,binarize_hearts,18)


print("Titanic:")
root1 = t.Node(train1,train_targets1)
root1.build_tree(root1)
t.classify_leaves(root1)
ex_train1 = t.classify_test(train1,root1)
c_train1 = t.results(ex_train1, train_targets1)
print("Train percentage: ",c_train1)
expected1 = t.classify_test(test1,root1)
correct1 = t.results(expected1,test_targets1)
print("Proportion correct:",correct1)

print("Heart Attack Risk:")
root2 = t.Node(train2, train_targets2)
root2.build_tree(root2)
t.classify_leaves(root2)
ex_train2 = t.classify_test(train2,root2)
c_train2 = t.results(ex_train2, train_targets2)
print("Train percentage: ",c_train2)
expected2 = t.classify_test(test2,root2)
correct2 = t.results(expected2, test_targets2)
print("Proportion correct:",correct2)

