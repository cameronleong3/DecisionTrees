#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 10 09:56:07 2021

@author: cameronleong
"""

import math as m
import sys
M = 13
labels = ["class1","class2","class3","male","child","adult","elderly",
          "siblings/spouses","parents/children","low fare","Cherbourg","Queenstown","Southampton"]
condition_path = []
conditions = [a for a in range(len(labels))]

class Node:
    
    def __init__(self, data, targets):

        self.parent = None
        self.left = None
        self.right = None
        self.data = data
        self.targets = targets
        self.leaf = 1
        self.lsize = 0
        self.rsize = 0
        self.child_condition = None
        self.end = 0
        self.result = None
   
    def sort_by_cond(self,cond):
        zeroes = []             #data points with cond = 0
        z_targets = []
        ones = []               #data points with cond = 1
        one_targets = []
        for i in range(len(self.data)):
            if self.data[i][cond] == 0:
                zeroes.append(self.data[i]) 
                z_targets.append(self.targets[i])
            else:
                ones.append(self.data[i])
                one_targets.append(self.targets[i])
        return zeroes,ones, z_targets, one_targets 
    
    
    def insert(self,cond): #inserts data as left and right children based on cond
        if self.leaf == 0:  #only insert at leaves
            return False
        zeroes,ones, zt, ot = self.sort_by_cond(cond)   #get data to be inserted
        global zeroes_ex
        zeroes_ex = zeroes
        self.leaf = 0
        self.left = Node(zeroes,zt)
        self.right = Node(ones,ot)
        self.left.parent = self 
        self.right.parent = self
        self.child_condition = cond
        self.lsize += 1
        self.rsize += 1
        return True
    
    def entropy(self): #calculates entropy of node
        p_pos = 0                           #store number of data points whose target value is 1
        p_neg = 0
        if len(self.data) == 0:
            return sys.maxsize 
        for i in range(len(self.data)):     #traverse remaining data points       
            if self.targets[i] == 1  :            #if target value is 1
                p_pos += 1
            else:
                p_neg += 1                  #target value is 0
        p_pos/=len(self.data)               #get proportion of positive values
        p_neg = 1-p_pos                     #get proportion of negative values
        if p_pos == 1 or p_neg == 1: 
            return 0
        #print("p_neg = ",p_neg)
        temp1 = -p_pos * m.log2(p_pos)
        temp2 = p_neg * m.log2(p_neg)
        E = temp1 - temp2 #entropy calculation
        #print("Entropy:",E)
        return E
    
    def info_gainz(self,cond):
        gain = self.entropy()       #entropy of the node's whole data
        path0 = []
        path1 = []
        targets0 = []
        targets1 = []
        for i in range(len(self.data)):    #calculate prop of pos values and entropy of the set of positive values
            if self.data[i][cond] == 0:    #append points with condition = 0 to path1 for entropy calculation
                path0.append(self.data[i])
                targets0.append(self.targets[i])
            else:
                path1.append(self.data[i])
                targets1.append(self.targets[i])
        size = len(path0)+len(path1)
        if len(path0)*len(path1) == 0:  #no information gain if data isn't split
            return 0
        left_path = Node(path0,targets0)    #create unattached nodes to calc entropy
        right_path = Node(path1,targets1)
        sum = (len(path0)/size)*left_path.entropy()  #sum = (prop of data with cond = 0) * entropy(data with cond = 0)
        sum += (len(path1)/size)*right_path.entropy() #sum += (prop of data with cond = 1) * entropy(data with cond = 1)
        gain -= sum #information gain calculation
        return gain
    
    def max_gain(self):
        max = 0
        cond = 0
        for i in range(len(conditions)):            #only check conditions not used yet
            temp = self.info_gainz(conditions[i])   #calc information gain for each condition
            if temp > max: 
                max = temp 
                cond = conditions[i]
        #print("Max gain: ",labels[cond])
        return cond,max


  
            
        
    def build_tree(self,root):  #need to keep track of root
        if self == None:
            print("error")
            return
        #if self.done() == True:
         #   return
        if self.leaf == 0: #internal nodes
            if (self == root) & (self.right.end == 1) & (self.left.end == 1): #tree is done
                return
            if self.right.end == 1 & self.left.end == 1:     #can't build from current node
                self.end = 1
                root.build_tree(root)
                
            if self.left.end == 1:              #left is dead end, right is open
                self.right.build_tree(root)
                
            if self.right.end == 1:             #left is open, right is dead end
                self.left.build_tree(root)
            else:                               #both are open, build on shorter side
                self.left.build_tree(root)

        cond,gain = self.max_gain()
        if gain == 0:
            self.end = 1
            return
        self.insert(cond)
        len_update(root)
       # print("used condition",labels[cond])
       # print("Gain: ",gain)
        condition_path.append(cond)
        self.left.build_tree(root)
        self.right.build_tree(root)

        
def len_update(node):
    if node is None:
        return 0 ;
    else :
        # Compute the depth of each subtree
        lDepth = len_update(node.left)
        rDepth = len_update(node.right)

        # Use the larger one
        if (lDepth > rDepth):
            return lDepth+1
        else:
            return rDepth+1
        
def printLeaves(node):
 
    # base case
    if node == None:
        return
    if node.leaf == 1:
        print("Node has size:",len(node.data))
       # print("leaf result:",node.result)
        print("targets",node.targets)
        x,y = node.max_gain()
        print("gain:",y)
        if node.parent != None:
            print("condition:",node.parent.child_condition)
    else:
        printLeaves(node.left)
        printLeaves(node.right)

        
def classify_leaves(node):
    if node == None:
        return
    if node.leaf == 1:                  #for each leaf node, calculate the mode
        node.result = calc_mode(node)   #then test data will be classified as whatever the mode is
    else:
        classify_leaves(node.left)      #recurse to each leaf
        classify_leaves(node.right)
    return

def calc_mode(node):
    zeroes = 0
    ones = 0
    for i in range(len(node.targets)):
        if node.targets[i] == 0:
            zeroes += 1
        else:
            ones += 1
    if zeroes > ones:
        return 0
    return 1
'''
def printInorder(root):
 
    if root:
 
        # First recurse on left child
        if root.leaf == 0:
            printInorder(root.left)
 
        # then print the data of node
        print(root.child_condition),
 
        # now recurse on right child
        if root.leaf == 0:
            printInorder(root.right)
'''
def classify_test(data,root):
    expected_results = [-1 for i in range(len(data))]
    for i in range(len(data)):
        node = root
        while node.leaf == 0:
            cond = node.child_condition
            if data[i][cond] == 0:
                #print("going left")
                node = node.left
            else:
               # print("going right")
                node = node.right
        expected_results[i] = node.result
    return expected_results

def results(expected_results,targets):
    correct_percent = 0
    for i in range(len(expected_results)):
        if expected_results[i] == targets[i]:
            correct_percent += 1
    return correct_percent/len(expected_results)
                
    
      
        
    
    
    
    
    