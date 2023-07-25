# -*- coding: utf-8 -*-
"""
LOW-G Decision Tree Models

Created on Mon Jul 10 18:27:50 2023
@author: Courtney Siffert
"""

import pickle
import pandas as pd
from sklearn.metrics import f1_score

# this is the path to your pickle file (should be the same location as CSVs)
path = r'D:\CSV'

# the below function verifies that the dataframe you are working with is the same shape as the anticipated dataframe
def test_dataframe_shape():
    # load the dataframe to be tested
    with open(r'D:/working_df.pkl', 'rb') as file:
        df = pickle.load(file)
    # Perform the shape validation
    assert df.shape == (258905, 118)
    return df

# working dataframe that has 'Label', 'Dynamic Rollover', 'LOW-G' as the final 3 columns
df = test_dataframe_shape().reset_index(drop=True)


#####################################################
#####################################################
# Drop dependent columns, train test split
#####################################################
#####################################################

from sklearn.model_selection import train_test_split

# drop independent columns
exclude_dependent_cols = ['Label','LOW-G','Dynamic Rollover']
imb_set_dr = df.drop(columns=exclude_dependent_cols)

# select y value (single column) -- one for Dynamic Rollover and one for LOW-G
imb_set_dr_y = df.loc[:, 'LOW-G']

# split the data into training and testing sets
x_train_dr, x_test_dr_new, y_train_dr, y_test_dr_new = train_test_split(imb_set_dr, imb_set_dr_y, test_size=0.2, stratify=imb_set_dr_y, random_state=42)


#####################################################
#####################################################
# IMB LEARN : Sampling (random undersampling and near miss sampling)
#####################################################
#####################################################

from imblearn.under_sampling import RandomUnderSampler
from imblearn.under_sampling import NearMiss

# perform random under-sampling on training set only
rus = RandomUnderSampler(random_state=690)
x_resampled_dr, y_resampled_dr = rus.fit_resample(x_train_dr, y_train_dr)

# perform NearMiss on training set only
nearmiss = NearMiss()
x_near_dr, y_near_dr = nearmiss.fit_resample(x_train_dr, y_train_dr)


#####################################################
#####################################################
## Decision Trees
#####################################################
#####################################################

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

#######################
# Random Undersampling Decision Tree
#######################

# decision tree classifier
dt_classifier = DecisionTreeClassifier()

# parameter grid for tuning
param_grid = {
    'max_depth': [None, 3, 5, 7, 10, 20],
    'min_samples_split': [2, 5, 10, 20, 30],
    'criterion': ['gini', 'entropy']
}

# grid search for tuning
grid_search = GridSearchCV(dt_classifier, param_grid, cv=10)
grid_search.fit(x_resampled_dr, y_resampled_dr)

# find the best hyperparameters
best_params = grid_search.best_params_

# decision tree using the best parameters from tuning
best_dt_classifier = DecisionTreeClassifier(**best_params)
best_dt_classifier.fit(x_resampled_dr, y_resampled_dr)

# predictions w/ test set
y_pred = best_dt_classifier.predict(x_test_dr_new)

# accuracy
accuracy = accuracy_score(y_test_dr_new, y_pred)
print("Accuracy:", accuracy)

# f1 score
f1_1 = f1_score(y_test_dr_new, y_pred)
print(f1_1)

# confusion matrix
cm = confusion_matrix(y_test_dr_new, y_pred)
print(cm)

# classification Report
print("RUS DECISION TREE Classification Report: LOW-G:  \n",classification_report(y_test_dr_new, y_pred))

# Print model parameters used
print("Best Parameters for Random Undersampling Decision Tree : LOW-G: \n",best_params)


#######################
# Near Miss Sampling Decision Tree
#######################

# decision tree classifier
dt_classifier2 = DecisionTreeClassifier()

# define the parameter grid for tuning
param_grid = {
    'max_depth': [None, 3, 5, 7, 10, 20],
    'min_samples_split': [2, 5, 10, 20, 30],
    'criterion': ['gini', 'entropy']
}

# grid search for tuning
grid_search2 = GridSearchCV(dt_classifier2, param_grid, cv=10)
grid_search2.fit(x_near_dr, y_near_dr)

# find the best hyperparameters
best_params2 = grid_search2.best_params_

# decision tree classifier using the best parameters from tuning
best_dt_classifier2 = DecisionTreeClassifier(**best_params2)
best_dt_classifier2.fit(x_near_dr, y_near_dr)

# predictions w/ test set
y_pred2 = best_dt_classifier2.predict(x_test_dr_new)

# accuracy
accuracy2 = accuracy_score(y_test_dr_new, y_pred2)
print("Accuracy:", accuracy2)

# f1 score
f1_2 = f1_score(y_test_dr_new, y_pred2)
print(f1_2)

# confusion matrix
cm2 = confusion_matrix(y_test_dr_new, y_pred2)
print(cm2)

# classification report
print("RUS DECISION TREE Classification Report: LOW-G \n",classification_report(y_test_dr_new, y_pred2))

# print the best model parameters used
print(best_params2)


#######################
# NO UNDERSAMPLING Decision Tree
#######################

# decision tree classifier
dt_classifier3 = DecisionTreeClassifier()

# parameter grid for tuning
param_grid = {
    'max_depth': [None, 3, 5, 7, 10, 20],
    'min_samples_split': [2, 5, 10, 20, 30],
    'criterion': ['gini', 'entropy']
}

# grid search for tuning
grid_search3 = GridSearchCV(dt_classifier, param_grid, cv=10)
grid_search3.fit(x_resampled_dr, y_resampled_dr)

# find the best hyperparameters
best_params3 = grid_search3.best_params_

# decision tree using the best parameters from tuning
best_dt_classifier3 = DecisionTreeClassifier(**best_params3)
best_dt_classifier3.fit(x_train_dr, y_train_dr)

# predictions w/ test set
y_pred3 = best_dt_classifier3.predict(x_test_dr_new)

# accuracy
accuracy3 = accuracy_score(y_test_dr_new, y_pred3)
print("Accuracy:", accuracy3)

# f1 score
f1_13 = f1_score(y_test_dr_new, y_pred3)
print(f1_13)

# confusion matrix
cm3 = confusion_matrix(y_test_dr_new, y_pred3)
print(cm3)

# classification Report
print("DECISION TREE Classification Report: LOW-G:  \n",classification_report(y_test_dr_new, y_pred3))

# Print model parameters used
print("Best Parameters for Decision Tree : LOW-G: \n",best_params3)


############################################################################
############################################################################
# Visualize Best Decision Tree Model
############################################################################
############################################################################

# visualization decision tree
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# get list of names for labeling decision tree visualization
feature_names2 = x_near_dr.columns.tolist()
print("Feature Names:", feature_names2)

#list of dependent variable names for labeling decision tree
class_names3 = '0','1'

# plot decision tree
plt.figure(figsize=(45, 12))
plot_tree(best_dt_classifier3, feature_names= feature_names2, class_names=class_names3 , filled=True)
plt.show()

############################################################################
############################################################################
# Visualize Tree Pruning
############################################################################
############################################################################

from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# find Pruning Path
path = dt_classifier3.cost_complexity_pruning_path(x_resampled_dr, y_resampled_dr)
alphas = path['ccp_alphas']
impurities = path['impurities']

# plot pruning path
plt.figure(figsize=(10, 6))
plt.plot(alphas, impurities, marker='o', drawstyle='steps-post')
plt.xlabel('Effective Alpha')
plt.ylabel('Total Impurity of Leaves')
plt.title('Pruning Path')
plt.show()

############################################################################
############################################################################
# MODEL COMPARISONS Accuracy
############################################################################
############################################################################

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# round accuracies
accuracy = round(accuracy,4)
accuracy2 = round(accuracy2,4)
accuracy3 = round(accuracy3,4)

# list of model names
model_names = ['RUS Decision Tree', 'NM Decision Tree', 'no sampling Decision Tree']
accuracies = [accuracy,  accuracy2, accuracy3]

# plot bar graph
plt.figure(figsize=(14, 6))
sns.barplot(x=np.arange(len(model_names)), y=accuracies, palette='mako')
plt.xticks(np.arange(len(model_names)), model_names)
plt.ylabel('Accuracy')
plt.xlabel('Models')
plt.title('Accuracy Comparison of Decision Tree Models : Low-G')
plt.xticks(rotation=80)
plt.ylim([0, 1])

# add numeric values to each bar
for i, value in enumerate(accuracies):
    plt.annotate(str(value), xy=(i, value), ha='center', va='top')
    
plt.show()


############################################################################
############################################################################
# MODEL COMPARISONS F1
############################################################################
############################################################################

# round f1 scores
f1_1 = round(f1_1,4)
f1_2 = round(f1_2,4)
f1_13 = round(f1_13,4)
f1_4 = 0.69

# list of model names
model_names = ['RUS Decision Tree', 'NM Decision Tree', 'No Sample Decision Tree'] 
fscore = [f1_1, f1_2, f1_13]

# plot bar graph
plt.figure(figsize=(14, 6))
sns.barplot(x=np.arange(len(model_names)), y=fscore, palette='mako')
plt.xticks(np.arange(len(model_names)), model_names)
plt.ylabel('F1 Score')
plt.xlabel('Models')
plt.title('F1 Comparison of Decision Tree Models : Low-G')
plt.xticks(rotation=80)
plt.ylim([0, 1])

# add numeric values to each bar
for i, value in enumerate(fscore):
    plt.annotate(str(value), xy=(i, value), ha='center', va='top')
    
plt.show()


################################################
################################################
## VARIABLE IMPORTANCE PLOT
################################################
################################################

# feature importances
feature_imp = best_dt_classifier3.feature_importances_

# create DF w/ feature names and importances
feature_imp_df = pd.DataFrame({'Feature': imb_set_dr.columns, 'Importance': feature_imp})

# sort features
feature_imp_df = feature_imp_df.sort_values(by='Importance', ascending=False)

# take only the top 20 important features
top_20_features = feature_imp_df.head(20)

# plot of top 20 feature importances
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=top_20_features, palette='viridis')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Top 20 Feature Importance Plot for Decision Tree Model : Low-G')
plt.show()


