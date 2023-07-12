# -*- coding: utf-8 -*-
"""
Created on Sat Jul  8 15:33:05 2023

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

from modeling_template import return_df


df = return_df()

print(df.columns)
## split test and training 
x = df.copy()
x = x.drop(columns=['LOW-G'])
y = pd.DataFrame(df['LOW-G'])
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.25, random_state=71)
   
## check number of records
print(x_test.shape[0]) #121868
print(x_train.shape[0]) #365602

## check value counts for y_train/y_test
unique, counts = np.unique(y_train, return_counts=True)
print(np.array((unique, counts)).T) # [(0: 364333), (1:1269)] (0.35%)
unique, counts = np.unique(y_test, return_counts=True)
print(np.array((unique, counts)).T) # [(0: 121415), (1:453)] (0.37%)

## Randomly oversample the minority class

ros = RandomOverSampler(random_state=71)
x_train_ros, y_train_ros = ros.fit_resample(x_train, y_train)

unique, counts = np.unique(y_train_ros, return_counts=True)
print(np.array((unique, counts)).T) #[(0: 364333), (1:364333)] (50%)

## SMOTE oversampling

smote = SMOTE(random_state=71)
x_train_smote, y_train_smote = smote.fit_resample(x_train, y_train)

unique, counts = np.unique(y_train_smote, return_counts=True)
print(np.array((unique, counts)).T) #[(0: 364333), (1:364333)] (50%)
 
## Randomly undersample the minority class

rus = RandomUnderSampler(random_state=71)
x_train_rus, y_train_rus= rus.fit_resample(x_train, y_train)

unique, counts = np.unique(y_train_rus, return_counts=True)
print(np.array((unique, counts)).T) #[(0: 1269), (1:1269)] (50%)

## Undersample to majority class with NearMiss

nearmiss = NearMiss(version=3)
x_train_nearmiss, y_train_nearmiss= nearmiss.fit_resample(x_train, y_train)

unique, counts = np.unique(y_train_nearmiss, return_counts=True)
print(np.array((unique, counts)).T) #[(0: 1269), (1:1269)] (50%)

## Standardize data

scaler = StandardScaler()
x_test = scaler.fit_transform(x_test)

x_train = scaler.fit_transform(x_train)
x_train_ros = scaler.fit_transform(x_train_ros)
x_train_smote = scaler.fit_transform(x_train_smote)
x_train_rus = scaler.fit_transform(x_train_rus)
x_train_nearmiss = scaler.fit_transform(x_train_nearmiss)

# Dimensionality Reduction

## PCA

pca = PCA(n_components = .85)
# x_train_pca = pca.fit_transform(x_train)
# x_train_ros_pca = pca.fit_transform(x_train_ros)
# x_train_smote_pca = pca.fit_transform(x_train_smote)
x_train_rus_pca = pca.fit_transform(x_train_rus)
x_train_nearmiss_pca = pca.fit_transform(x_train_nearmiss)


pc_values = np.arange(pca.n_components_)+1
plt.plot(pc_values, pca.explained_variance_ratio_, 'o-', linewidth=2, color='blue')
plt.title('Scree Plot')
plt.xlabel('Principal Component')
plt.ylabel('Variance Explained')
plt.show()

print(pca.explained_variance_ratio_)

x_test_pca = pca.transform(x_test)

## Kernel PCA

kpca = KernelPCA(n_components=15, kernel='rbf',
                  gamma = 1, random_state = 71)

x_nearmiss_kpca = kpca.fit_transform(x_train_nearmiss)
x_trian_kpca = x_nearmiss_kpca

x_test_kpca = kpca.transform(x_test)

kpca_explained_variance = np.var(x_nearmiss_kpca, axis=0)
kpca_explained_variance_ratio = kpca_explained_variance / np.sum(kpca_explained_variance)
np.cumsum(kpca_explained_variance_ratio)

## t-SNE

tsne = TSNE()

x_train_rus_tsne = tsne.fit_transform(x_train_rus_pca)
x_train_nearmiss_tsne = tsne.fit_transform(x_train_nearmiss_pca)

x_test_tsne = tsne.transform(x_test_pca)

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

x_train_grid = x_train_nearmiss_pca
y_train_grid = y_train_nearmiss
x_test_grid = x_test_pca
y_test_grid = y_test

param_grid = {'C': [0.01, 0.1, 0.5, 1, 10, 100], 
              'gamma': [1, 0.75, 0.5, 0.25, 0.1, 0.01, 0.001], 
              'kernel': ['rbf', 'poly', 'linear']} 

start = time.time()
print("--- %s seconds ---" % (time.time() - start))
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=1, cv=5)
grid.fit(x_train_grid, y_train_grid.values.ravel())
end = time.time()
print("The time of execution of above program is :",
      (end-start), "s")
best_params = grid.best_params_
svm_clf = SVC(**best_params)
svm_clf.fit(x_train_grid, y_train_grid.values.ravel())
print_score(svm_clf, x_train_grid, y_train_grid, x_test_grid, y_test_grid, train=True)
print_score(svm_clf, x_train_grid, y_train_grid, x_test_grid, y_test_grid, train=False)
best_params

# ## Linear Kernel SVM

svm = SVC(kernel='linear', gamma=1, C=100)
svm.fit(x_train, y_train.values.ravel())
print_score(svm, x_train, y_train, x_test, y_test, train=True)
print_score(svm, x_train, y_train, x_test, y_test, train=False)

# ## Polynomial Kernel SVM

# model = SVC(kernel='poly', degree=2, gamma='auto', coef0=1, C=5)
# model.fit(x_train_ros, y_train_ros.values.ravel())
# print_score(model, x_train_ros, y_train_ros, x_test, y_test, train=True)
# print_score(model, x_train_ros, y_train_ros, x_test, y_test, train=False)

# ## Radial Kernel SVM

model = SVC(kernel='rbf', gamma=0.25, C=10)
model.fit(x_train_grid, y_train_grid.values.ravel())
print_score(model, x_train_grid, y_train_grid, x_test_grid, y_test_grid, train=True)
print_score(model, x_train_grid, y_train_grid, x_test_grid, y_test_grid, train=False)


