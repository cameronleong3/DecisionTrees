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
N = 100

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
    for i in range(len(x)):
        z[i][18] = x[i][13]    #target values
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
            
 #MAIN
data1 = binarize_titanic(raw_data1)       #convert data into binary categories
data2 = binarize_hearts(raw_data2)
val_data1 = r.choices(data1, k = round(.3*len(data1)))    #set aside data for validation
data1 =  r.choices(data1, k = round(.7*len(data1)))
val_data2 = r.choices(data2, k = round(.3*len(data2)))    #set aside data for validation
data2 =  r.choices(data2, k = round(.7*len(data2)))

def random_forest(data,targets_idx):
    
    test_size = round(.3*len(data))         #use 30% of data to test
    train_size = round(.7*len(data))     #use 70% of the data to train
    test = r.choices(data, k = test_size)    #get test data
    test_targets = []
    for i in range(test_size):                 #save test targets
        test_targets.append(test[i][targets_idx])
    test = np.delete(test,(targets_idx - 1),axis = 1)
    
    expected_array = np.zeros((N,len(test)))
    for i in range(N):
        print("iteration: ",i)
        train = r.choices(data,k = train_size)  
        train_targets = []
        for j in range(train_size):
            train_targets.append(train[j][targets_idx])  #target values in column 14
        train = np.delete(train,(targets_idx - 1),axis = 1)    #remove targets from data
        root = t.Node(train,train_targets)     #create root 
        root.build_tree(root)                   #build tree
        t.classify_leaves(root)                  #classify leaves of tree
        expected = t.classify_test(test,root)  
        expected_array[i] = expected            #save expected results from testing on this tree
    
    final_expected = np.zeros(len(test))
    for i in range(len(test)):         #calculate mode expected for each data point
        zs = 0
        ones = 0
        for j in range(N):
            if expected_array[j][i] == 0: #target i in iteration j
                zs += 1                     #find if more results are 0 or 1
            else:
                ones += 1
        if ones > zs:                   #otherwise its zero already
            final_expected[i] = 1
    return final_expected, test_targets

print("Titanic:")
expected,actual = random_forest(data1,14)
t_ex, t_actual = random_forest(val_data1,14)
correct = t.results(t_ex,t_actual)
print("train:",correct)
correct = t.results(expected,actual)
print("test:",correct)   

print("Heart Attacks:")
expected,actual = random_forest(data2,18)
t_ex, t_actual = random_forest(val_data2,18)
correct = t.results(t_ex, t_actual)
print("train:",correct)
correct = t.results(expected,actual)
print("Percentage correct:",correct)   
    
