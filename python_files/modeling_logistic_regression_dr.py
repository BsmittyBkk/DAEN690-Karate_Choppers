# -*- coding: utf-8 -*-
"""
Dynamic Rollover Logistic Regression Models

Created on Mon Jul 10 18:27:50 2023
@author: Courtney Siffert
"""

from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import RandomUnderSampler
import sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
from sklearn.metrics import precision_recall_curve, auc


# this is the path to your pickle file (should be the same location as CSVs)
path = r'D:\CSV'

# the below function verifies that the dataframe you are working with is the same shape as the anticipated dataframe

def test_dataframe_shape():
    # load the dataframe to be tested
    with open(r'D:/CSV/dynamic_rollover.pkl', 'rb') as file:
        df = pickle.load(file)
    # Perform the shape validation
    #assert df.shape == (258905, 118)
    return df


# working dataframe that has 'Label', 'Dynamic Rollover', 'LOW-G' as the final 3 columns
df = test_dataframe_shape().reset_index(drop=True)


#####################################################
#####################################################
# Drop dependent columns, StandardScaler, train test split
#####################################################
#####################################################

# drop independent columns
exclude_dependent_cols = ['Dynamic Rollover']
imb_set_dr = df.drop(columns=exclude_dependent_cols)

# create a Standard Scaler object
scaler = StandardScaler()

# fit the scaler to your dataset
scaler.fit(imb_set_dr)

# transform dataset
normalized_data = scaler.transform(imb_set_dr)

# create a new DataFrame for norm data
imb_set_dr = pd.DataFrame(normalized_data, columns=imb_set_dr.columns)

# select y value (single column) -- one for Dynamic Rollover
imb_set_dr_y = df.loc[:, 'Dynamic Rollover']

# split the data into training and testing sets
x_train_dr, x_test_dr_new, y_train_dr, y_test_dr_new = train_test_split(
    imb_set_dr, imb_set_dr_y, test_size=0.25, stratify=imb_set_dr_y, random_state=42)


#####################################################
#####################################################
# IMB LEARN : Sampling (random undersampling to combat imbalanced dataset)
#####################################################
#####################################################

# perform random under-sampling on training set only
rus = RandomUnderSampler(random_state=42)
x_resampled_dr, y_resampled_dr = rus.fit_resample(x_train_dr, y_train_dr)


#####################################################
#####################################################
# IMB LEARN : Sampling (random undersampling to combat imbalanced dataset)
#####################################################
#####################################################


# Assuming X is your feature matrix
pca = PCA()
pca.fit(x_resampled_dr)

# Plot the cumulative explained variance ratio
explained_variance_ratio_cumulative = np.cumsum(pca.explained_variance_ratio_)
plt.plot(explained_variance_ratio_cumulative)
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance Ratio')
plt.title('Cumulative Explained Variance : Dynamic Rollover')
plt.grid(True)
plt.show()


explained_variance = pca.explained_variance_

# Plot the explained variance
plt.plot(range(1, len(explained_variance) + 1), explained_variance)
plt.xlabel('Number of Components')
plt.ylabel('Explained Variance')
plt.title('Explained Variance Elbow Plot : Dynamic Rollover')
plt.grid(True)
plt.show()


#  cross validation to find the most efficient number of principal components to use
components_range = range(1, len(x_resampled_dr.columns) + 1)
cv_scores = []

for n_components in components_range:
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(x_resampled_dr)
    model = LogisticRegression()
    cv_score = cross_val_score(
        model, X_train_pca, y_resampled_dr, cv=5, scoring='accuracy').mean()
    cv_scores.append(cv_score)

best_n_components = components_range[np.argmax(cv_scores)]
print(best_n_components)


# Specify the number of principal components to retain
pca = PCA(n_components=14)
X_train_pca = pca.fit_transform(x_resampled_dr)
X_test_pca = pca.transform(x_test_dr_new)

logreg4 = LogisticRegression()
logreg4.fit(X_train_pca, y_resampled_dr)

y_pred4 = logreg4.predict(X_test_pca)

accuracy_pca10 = accuracy_score(y_test_dr_new, y_pred4)
print("Accuracy:", accuracy_pca10)

f1_4 = f1_score(y_test_dr_new, y_pred4)
print(f1_4)


#####################################################
#####################################################
# Logistic Regression basic model
#####################################################
#####################################################

# logistic regression model
logistic_regression = LogisticRegression(random_state=42)

# hyperparameter grid for tuning
param_grid = {
    'penalty': ['l1', 'l2', 'None'],
    'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
    'solver': ['liblinear', 'saga', 'lbfgs'],
    'max_iter': [10000000],
    'class_weight': ['balanced', 'None']
}

# grid search for hyperparameter tuning
grid_search = GridSearchCV(
    logistic_regression, param_grid, cv=5, scoring='roc_auc', verbose=3)

# get a list of scoring names to use in the grid search above
sklearn.metrics.get_scorer_names()

# hyperparameter tuning with tqdm for progress display
with tqdm(total=len(param_grid['penalty']) * len(param_grid['C']) * len(param_grid['solver']) * len(param_grid['max_iter']) * len(param_grid['class_weight'])) as pbar:
    for params in grid_search.param_grid:
        grid_search.fit(X_train_pca, y_resampled_dr)
        pbar.update(1)

# best hyperparameters from grid search
best_params = grid_search.best_params_

# new logistic regression model with best hyperparameters
best_logistic_regression = LogisticRegression(**best_params)

# train the model
best_logistic_regression.fit(X_train_pca, y_resampled_dr)

# predictions with the test set
y_pred_done = best_logistic_regression.predict(X_test_pca)  # (x_test_dr_new)

# Print model parameters used
print("Best Parameters for Logistic Regression Dynamic Rollover Model:  \n", best_params)

# Print classification report
print("Classification Report for Logistic Regression Dynamic Rollover Model:  \n",
      (classification_report(y_test_dr_new, y_pred_done)))

# Print confusion matrix
print("Classification Report for Logistic Regression Dynamic Rollover Model:  \n",
      (confusion_matrix(y_test_dr_new, y_pred_done)))

# model evaluations
print("Accuracy:", round(accuracy_score(y_test_dr_new, y_pred_done), 4))
print("Recall:", round(recall_score(y_test_dr_new, y_pred_done), 4))
print("Precision:", round(precision_score(y_test_dr_new, y_pred_done), 4))
print("F1:", round(f1_score(y_test_dr_new, y_pred_done), 4))

# model information
print(best_logistic_regression.solver)  # lbfgs
print(best_logistic_regression.n_iter_)  # 129
print(best_logistic_regression.C)


############################################################################
############################################################################
# VARIABLE IMPORTANCE PLOT
############################################################################
############################################################################

# find weights of dependent variables
coefs = best_logistic_regression.coef_[0]

# absolute value of the weights
abs_coefs = np.abs(coefs)

# array of variable names
variable_names = np.array(x_train_dr.columns)

# sort coefficients and variable names descending
sorted_indices = np.argsort(abs_coefs)[::-1]
sorted_coefs = abs_coefs[sorted_indices]
sorted_variable_names = variable_names[sorted_indices]

# select the top 20 most important variables
top_10_coefs = sorted_coefs[:20]
top_10_variable_names = sorted_variable_names[:20]

# bar plot of variable importance
plt.figure(figsize=(10, 6))
sns.barplot(x=np.arange(len(top_10_coefs)), y=top_10_coefs, palette='viridis')
plt.xticks(range(len(top_10_coefs)),
           top_10_variable_names, rotation='vertical')
plt.xlabel('Variables')
plt.ylabel('Absolute Coefficients')
plt.title(
    'Variable Importance Plot : Dynamic Rollover Logistic Regression Model')
plt.tight_layout()
plt.show()

############################################################################
############################################################################
# Precision Recall Curve For best performing model
############################################################################
############################################################################

# calc precision-recall curve
precision, recall, _ = precision_recall_curve(y_test_dr_new, y_pred_done)

# calc AUC for the precision-recall curve
pr_auc = auc(recall, precision)

# plot precision-recall curve
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, color='b', label=f'PR AUC = {pr_auc:.2f}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve for Dynamic Rollover Logistic Regression')
plt.legend(loc='lower left')
plt.grid(True)
plt.show()


############################################################################
############################################################################
# Research into variable linearity to explain poor model performance
############################################################################
############################################################################


# correlation matrix
correlation_matrix = df.corr()
correlation_matrix.fillna(0.00, inplace=True)
print(correlation_matrix)

# heatmap
plt.figure(figsize=(15, 6))
sns.heatmap(correlation_matrix, annot=True,
            cmap='coolwarm', vmin=-1, vmax=1, fmt='.2f')
plt.title('Correlation Matrix : Dynamic Rollover')
plt.show()

# corr with the dependent variable
correlation_with_Y = correlation_matrix['Dynamic Rollover']

# abs corr in desc order
absolute_correlations = correlation_with_Y.abs().sort_values(ascending=False)
print(absolute_correlations)

# PCA corr
principal_df = pd.DataFrame(data=X_test_pca, columns=[
                            'PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8', 'PC9', 'PC10', 'PC11', 'PC12', 'PC13', 'PC14'])
combined_df = pd.concat([principal_df, y_resampled_dr], axis=1)

correlation_matric2 = combined_df.corr()

plt.figure(figsize=(15, 6))
sns.heatmap(correlation_matric2, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Matrix : Dynamic Rollover PCA')
plt.show()

# PCA corr with dependent variable
correlation_with_Y2 = correlation_matric2['Dynamic Rollover']

# abs corr in desc order
absolute_correlations2 = correlation_with_Y2.abs().sort_values(ascending=False)
print(absolute_correlations2)


