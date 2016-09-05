# -*- coding: utf-8 -*-
"""
Created on Sat Jul 30 19:35:42 2016

@author: sherlock
"""

import os
import re
import pandas as pd

trainDataPath = '/Users/sherlock/Machinelearning/utf-8'
outTrainFile = 'trainCleaned.csv'



def cleanTrain(trainDataPath,outTrainFile):
    print 'start cleaning train data..'
    train_file_names=os.listdir(trainDataPath)
    #print train_file_names
    train_data_list = []
    for train_file_name in train_file_names:
        if not train_file_name.endswith('.txt'):
            continue
        print train_file_name
        print train_file_name[0]
        print type(train_file_name[0])
        train_file = os.path.join(trainDataPath,train_file_name)
        
        label = int(train_file_name[0])
        
        with open (train_file,'r') as f:
            lines = f.read().splitlines()
            
        labels = [label]*len(lines)
        
        labels_series = pd.Series(labels)
        lines_series = pd.Series(lines)
        
        data_pd = pd.concat([labels_series,lines_series],axis =1)
        train_data_list.append(data_pd)
        
    train_data_pd = pd.concat(train_data_list,axis = 0)
    
    train_data_pd.columns = ['label','text']
    train_data_pd.to_csv(os.path.join(trainDataPath,outTrainFile),index = None,\
    encoding = 'utf-8',header = True)
    
cleanTrain(trainDataPath,outTrainFile)



test_data_path = '/Users/sherlock/Machinelearning/test'
test_file_name = 'emotionTest.csv' 
out_test_file_name = 'theTest.csv'

def clean_test(test_data_path,test_file_name,out_test_file_name):
    print 'cleaning the test data now...'
    test_file = os.path.join(test_data_path, test_file_name)
    print test_file
    with open (test_file,'r') as f:
        lines = f.read().splitlines()
    
    lines_series = pd.Series(lines)
    
    test_data_list = pd.Series(lines_series, name = 'text')
    
    test_data_list.to_csv(os.path.join(test_data_path,out_test_file_name),index \
     = None, encoding = 'utf-8',header = True)
    
clean_test(test_data_path,test_file_name,out_test_file_name)


