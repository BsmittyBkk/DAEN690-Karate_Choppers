# -*- coding: utf-8 -*-
"""
Created on Sat Jul 22 15:43:52 2023

@author: jetow
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler, NearMiss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.svm import LinearSVC, SVC
from sklearn.model_selection import GridSearchCV
import time

from numpy import mean
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold

print(df_dr.columns)
## split test and training 
x = df_dr.copy()
x = x.drop(columns=['label_Dynamic Rollover'])
y = pd.DataFrame(df_dr['label_Dynamic Rollover'])
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.25, random_state=42)
   
## check number of records
print(x_test.shape[0])
print(x_train.shape[0])

## check value counts for y_train/y_test
unique, counts = np.unique(y_train, return_counts=True)
print(np.array((unique, counts)).T)
unique, counts = np.unique(y_test, return_counts=True)
print(np.array((unique, counts)).T)

## Standardize data

scaler = StandardScaler()
x_test = scaler.fit_transform(x_test)
x_train = scaler.fit_transform(x_train)

# SVM

## Define metrics

def print_score(clf, x_train, y_train, x_test, y_test, train=True):
    if train:
        pred = clf.predict(x_train)
        clf_report = pd.DataFrame(classification_report(y_train, pred, output_dict=True))
        print("Train Result:\n ================================================")
        print(f"Accuracy Score: {accuracy_score(y_train, pred) * 100:.2f}%")
        print("_______________________________________________")
        print(f"CLASSIFICATION REPORT:\n {clf_report}")
        print("_______________________________________________")
        print(f"Confusion Matrix: \n {confusion_matrix(y_train, pred)}\n")
    elif train==False:
        pred = clf.predict(x_test)
        clf_report = pd.DataFrame(classification_report(y_test, pred, output_dict=True))
        print("Test Result:\n ================================================")        
        print(f"Accuracy Score: {accuracy_score(y_test, pred) * 100:.2f}%")
        print("_______________________________________________")
        print(f"CLASSIFICATION REPORT:\n {clf_report}")
        print("_______________________________________________")
        print(f"Confusion Matrix: \n {confusion_matrix(y_test, pred)}\n")



## Hyperparameter Tuning

x_train = x_train
y_train = y_train
x_test = x_test
y_test = y_test

param_grid = {'C': [0.1, 1, 10, 100, 1000], 
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
              'kernel': ['rbf']} 

start = time.time()
print("--- %s seconds ---" % (time.time() - start))
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=3)
grid.fit(x_train, y_train.values.ravel())
end = time.time() 
print("The time of execution of above program is :",
      (end-start), "s")
best_params = grid.best_params_
svm_clf = SVC(**best_params)
svm_clf.fit(x_train, y_train.values.ravel())
print_score(svm_clf, x_train, y_train, x_test, y_test, train=True)
print_score(svm_clf, x_train, y_train, x_test, y_test, train=False)
best_params

# ## Linear Kernel SVM

# svm = SVC(kernel='linear', gamma=1, C=100)
# svm.fit(x_train, y_train.values.ravel())
# print_score(svm, x_train, y_train, x_test, y_test, train=True)
# print_score(svm, x_train, y_train, x_test, y_test, train=False)

# ## Polynomial Kernel SVM

# model = SVC(kernel='poly', degree=2, gamma='auto', coef0=1, C=5)
# model.fit(x_train_ros, y_train_ros.values.ravel())
# print_score(model, x_train_ros, y_train_ros, x_test, y_test, train=True)
# print_score(model, x_train_ros, y_train_ros, x_test, y_test, train=False)

# ## Radial Kernel SVM

model = SVC(kernel='rbf', gamma=1, C=1000, class_weight='balanced')
model.fit(x_train, y_train.values.ravel())
print_score(model, x_train, y_train, x_test, y_test, train=True)
print_score(model, x_train, y_train, x_test, y_test, train=False)


# # define model
# model = SVC(gamma='scale')

# # define evaluation procedure
# cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

# # evaluate model
# scores = cross_val_score(model, X, y, scoring='roc_auc', cv=cv, n_jobs=-1)
# # summarize performance
# print('Mean ROC AUC: %.3f' % mean(scores))