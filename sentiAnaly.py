# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 18:34:55 2016

@author: sherlock
"""

import jieba 
import time
start_time = time.time()
import re
import os
import nltk
import pandas as pd


import numpy as np

from sklearn.metrics import precision_recall_curve, roc_curve, auc
from sklearn.cross_validation import ShuffleSplit

#from cleandata import clean_train,clean_test

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import f1_score
from sklearn.base import BaseEstimator
from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

def words_to_features(raw_line,stopwords_path='/Users/sherlock/Machinelearning/cnstopwords.txt'):
    stopwords = {}.fromkeys([line.rstrip() for line in open (stopwords_path)])
    chinese_only = raw_line
    words_lst = jieba.cut(chinese_only)
    meaninful_words = []
    for word in words_lst:
        word = word.encode('utf8')
        if word not in stopwords:
            meaninful_words.append(word)
        return ' '.join(meaninful_words)

train_data_path='/Users/sherlock/Machinelearning/utf-8'
train_file_name='trainCleaned.csv'
test_data_path = '/Users/sherlock/Machinelearning/test'
test_file_name = 'theTest.csv'


      
def drawfeature(train_data_path,train_file_name,test_data_path,test_file_name):
    train_file = os.path.join(train_data_path,train_file_name)
    train_data = pd.read_csv(train_file)
    n_data_train = train_data['text'].size
    print 'n_data_train is %s' %n_data_train
    print type(n_data_train)
    
    test_file = os.path.join(test_data_path,test_file_name)
    test_data = pd.read_csv(test_file)
    n_data_test = test_data['text'].size
    print 'n_data_test is %s' %n_data_test
    print type(n_data_test)
    
    vectorizer = CountVectorizer(analyzer='word',tokenizer = None,
        preprocessor = None, stop_words=None, max_features = 5000)
    transformer = TfidfTransformer()
    
    train_data_words = []
    
    print 'start with words in train data set'
    for i in xrange(n_data_train):
        if((i+1)%1000 == 0):
            print 'Drawfeatures line %d of %d' %(i+1,n_data_train)
        train_data_words.append(words_to_features(train_data['text'][i]))
    print 'start bag of words in train data....'
    train_data_features = vectorizer.fit_transform(train_data_words)
    train_data_features = train_data_features.toarray()
    print 'start tfidf in train data....'
    train_data_features = transformer.fit_transform(train_data_features)
    train_data_features = train_data_features.toarray()
    #test-data processing
    test_data_words = []
    for i in xrange(n_data_test):
        if((i+1)%1000 == 0):
            print 'Drawfeatures line %d of %d' %(i+1,n_data_test)
        test_data_words.append(words_to_features(test_data['text'][i]))
    
    test_data_features = vectorizer.fit_transform(test_data_words)
    test_data_features = test_data_features.toarray()
    
    
       
    print'randome forest go...'
    forest = RandomForestClassifier(n_estimators = 13)
    forest = forest.fit(train_data_features,train_data['label'])
    pred = forest.predict(test_data_features)
    pred = pd.Series(pred,name='Target')
    pred.to_csv('SENTI_RF.CSV',index=None, header = None)

    
    print'naive baby go...'
    mnb = MultinomialNB(alpha=0.01)
    mnb = mnb.fit(train_data_features,train_data['label'])
    pred = mnb.predict(test_data_features)
    pred = pd.Series(pred,name = 'Target')
    pred.to_csv('SENTI_MNB',index = None, header = True)

train_data_path='/Users/sherlock/Machinelearning/utf-8'
train_file_name='trainCleaned.csv'
test_data_path = '/Users/sherlock/Machinelearning/test'
test_file_name = 'theTest.csv'    
drawfeature(train_data_path,train_file_name,test_data_path,test_file_name)
    
        
    
    
    
        




