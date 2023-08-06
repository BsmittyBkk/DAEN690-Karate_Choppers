import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score, confusion_matrix, classification_report
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import time
from pathlib import Path


# this is the path to your pickle file (should be the same location as CSVs)
path = Path('../../CSV')

with open(f'{path}/low_g.pkl', 'rb') as file:
    df = pickle.load(file)

# drop index
df = df.reset_index(drop=True)
# define independent variables and dependent variable
X = df.drop('LOW-G', axis=1)
y = df['LOW-G']

# create training set and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

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
      (end-start), "m")
best_params = grid.best_params_
svm_clf = SVC(**best_params)
svm_clf.fit(x_train, y_train.values.ravel())
print_score(svm_clf, x_train, y_train, x_test, y_test, train=True)
print_score(svm_clf, x_train, y_train, x_test, y_test, train=False)
best_params


# ## Polynomial Kernel SVM

# model = SVC(kernel='poly', degree=2, gamma='auto', coef0=1, C=5)
# model.fit(x_train_ros, y_train_ros.values.ravel())
# print_score(model, x_train_ros, y_train_ros, x_test, y_test, train=True)
# print_score(model, x_train_ros, y_train_ros, x_test, y_test, train=False)

## Radial Kernel SVM

model = SVC(kernel='rbf', gamma=1, C=1000, class_weight='balanced', probability=True, random_state=42)
model.fit(x_train, y_train.values.ravel())
print_score(model, x_train, y_train, x_test, y_test, train=True)
print_score(model, x_train, y_train, x_test, y_test, train=False)

    
## Visualizations: hyperparameters

# C Values

# define lists to collect scores
train_f1_scores, test_f1_scores = list(), list()
train_recall_scores, test_recall_scores = list(), list()
train_prec_scores, test_prec_scores = list(), list()

# define list of C values to plot
c_values = [0.1, 1, 10, 100, 1000]

for c in c_values:
    # configure the model
    model = SVC(kernel='rbf', gamma=1, C=c, class_weight='balanced', random_state=42)
    # fit the training dataset
    model.fit(x_train, y_train.values.ravel())
    # evaluate training dataset
    train_yhat = model.predict(x_train)
    train_f1 = f1_score(y_train, train_yhat)
    train_recall = recall_score(y_train, train_yhat)
    train_prec = precision_score(y_train, train_yhat)
    train_f1_scores.append(train_f1)
    print(train_f1_scores)
    train_recall_scores.append(train_recall)
    print(train_recall_scores)
    train_prec_scores.append(train_prec)
    print(train_prec_scores)
    # evaluate test dataset
    test_yhat = model.predict(x_test)
    test_f1 = f1_score(y_test, test_yhat)
    test_recall = recall_score(y_test, test_yhat)
    test_prec = precision_score(y_test, test_yhat)
    test_f1_scores.append(test_f1)
    print(test_f1_scores)
    test_recall_scores.append(test_recall)
    print(test_recall_scores)
    test_prec_scores.append(test_prec)
    print(test_prec_scores)    
    # summarize progress
    print('f1: >%d, train: %.3f, test: %.3f' % (c, train_f1, test_f1))
    print('recall: >%d, train: %.3f, test: %.3f' % (c, train_recall, test_recall))
    print('precision: >%d, train: %.3f, test: %.3f' % (c, train_prec, test_prec))

train_f1_scores
test_f1_scores

# plot of train and test scores vs tree depth
plt.plot(c_values, test_f1_scores, '-o', label='f1 score')
plt.plot(c_values, test_recall_scores, '-o', label='recall')
plt.plot(c_values, test_prec_scores, '-o', label='precision')
plt.title("Dynamic Rollover: C Values")
plt.legend()
plt.xlabel("C Value (Gamma = 1)")
plt.show()

# Gamma Values

# define lists to collect scores
train_f1_scores, test_f1_scores = list(), list()
train_recall_scores, test_recall_scores = list(), list()
train_prec_scores, test_prec_scores = list(), list()

# define list of C values to plot
gamma_values = [1, 0.1, 0.01, 0.001, 0.0001]

for g in gamma_values:
    # configure the model
    model = SVC(kernel='rbf', gamma=g, C=1000, class_weight='balanced', random_state=42)
    # fit the training dataset
    model.fit(x_train, y_train.values.ravel())
    # evaluate training dataset
    train_yhat = model.predict(x_train)
    train_f1 = f1_score(y_train, train_yhat)
    train_recall = recall_score(y_train, train_yhat)
    train_prec = precision_score(y_train, train_yhat)
    train_f1_scores.append(train_f1)
    print(train_f1_scores)
    train_recall_scores.append(train_recall)
    print(train_recall_scores)
    train_prec_scores.append(train_prec)
    print(train_prec_scores)
    # evaluate test dataset
    test_yhat = model.predict(x_test)
    test_f1 = f1_score(y_test, test_yhat)
    test_recall = recall_score(y_test, test_yhat)
    test_prec = precision_score(y_test, test_yhat)
    test_f1_scores.append(test_f1)
    print(test_f1_scores)
    test_recall_scores.append(test_recall)
    print(test_recall_scores)
    test_prec_scores.append(test_prec)
    print(test_prec_scores)    
    # summarize progress
    print('f1: >%d, train: %.3f, test: %.3f' % (g, train_f1, test_f1))
    print('recall: >%d, train: %.3f, test: %.3f' % (g, train_recall, test_recall))
    print('precision: >%d, train: %.3f, test: %.3f' % (g, train_prec, test_prec))


# plot of train and test scores vs tree depth
plt.plot(gamma_values, test_f1_scores, '-o', label='f1 score')
plt.plot(gamma_values, test_recall_scores, '-o', label='recall')
plt.plot(gamma_values, test_prec_scores, '-o', label='precision')
plt.legend()
plt.title("Dynamic Rollover: Gamma Values")
plt.xlabel("Gamma Value (C = 1000)")
plt.show()

## Precision-Recall Curve

from sklearn.metrics import precision_recall_curve

# Predict probability
y_prob_train = model.predict_proba(x_train)[:,1]
y_prob_test = model.predict_proba(x_test)[:,1]

precision, recall, thresholds = precision_recall_curve(y_train, y_prob_train)
plt.fill_between(recall, precision)
plt.ylabel("Precision")
plt.xlabel("Recall")
plt.title("Train Precision-Recall curve")

plt.fill_between(recall, precision)
plt.ylabel("Precision")
plt.xlabel("Recall")
plt.title("Train Precision-Recall curve")
