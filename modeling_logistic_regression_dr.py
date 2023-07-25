# -*- coding: utf-8 -*-
"""
LOW-G Logistic Regression Models

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
# Drop dependent columns, StandardScaler, train test split
#####################################################
#####################################################

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# drop independent columns
exclude_dependent_cols = ['Label','LOW-G','Dynamic Rollover']
imb_set_dr = df.drop(columns=exclude_dependent_cols)

# create a Standard Scaler object
scaler = StandardScaler()

# fit the scaler to your dataset
scaler.fit(imb_set_dr)

# transform dataset
normalized_data = scaler.transform(imb_set_dr)

# create a new DataFrame for norm data
imb_set_dr = pd.DataFrame(normalized_data, columns=imb_set_dr.columns)

# select y value (single column) -- one for LOW-G
imb_set_dr_y = df.loc[:, 'Dynamic Rollover']

# split the data into training and testing sets
x_train_dr, x_test_dr_new, y_train_dr, y_test_dr_new = train_test_split(imb_set_dr, imb_set_dr_y, test_size=0.2, stratify=imb_set_dr_y, random_state=690)


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
# Logistic Regression basic model
#####################################################
#####################################################

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# create a Logistic Regression model
logreg = LogisticRegression(max_iter=1000)

# train the model
logreg.fit(x_resampled_dr, y_resampled_dr)

# make predictions w/ Test Data
y_pred_imb_dr = logreg.predict(x_test_dr_new)

# accuracy score
accuracy = accuracy_score(y_test_dr_new, y_pred_imb_dr)
print("Accuracy:", round(accuracy,4))

# confusion matrix
cm = confusion_matrix(y_test_dr_new, y_pred_imb_dr)
print(cm)

# classification report
print(classification_report(y_test_dr_new, y_pred_imb_dr))

#f1_score
f1_1 = f1_score(y_test_dr_new, y_pred_imb_dr)
print(f1_1)

#model information
print(logreg.solver) #lbfgs
print(logreg.n_iter_) #129
#print(logreg.C_[0])

#####################################################
#####################################################
# RIDGE Liblinear Random Under Sampling
#####################################################
#####################################################

from sklearn.linear_model import LogisticRegressionCV

# create and fit the RIDGE logistic regression model with Randum Undersampling
logreg3 = LogisticRegressionCV(Cs=10, cv=5, penalty='l2', solver='liblinear', random_state=42, max_iter=1000)
logreg3.fit(x_resampled_dr, y_resampled_dr)

# make predictions w/ test set
y_pred3 = logreg3.predict(x_test_dr_new)

# confusion matrix
cm = confusion_matrix(y_test_dr_new, y_pred3)
print(cm)

# classification Report
print("LIBLINEAR RIDGE RUS Classification Report: \n",classification_report(y_test_dr_new, y_pred3))

# f1 score
f1_3 = f1_score(y_test_dr_new, y_pred3)
print(f1_3)

# accuracy
accuracy3 = accuracy_score(y_test_dr_new, y_pred3)
print("Accuracy:", accuracy3)

#####################################################
#####################################################
# RIDGE LLBFGS Random Under Sampling
#####################################################
#####################################################

from sklearn.linear_model import LogisticRegressionCV

# create and fit the RIDGE lbfgs logistic regression model with Randum Undersampling
logregz = LogisticRegressionCV(Cs=10, cv=5, penalty='l2', solver='lbfgs', random_state=42, max_iter=1000)
logregz.fit(x_resampled_dr, y_resampled_dr)

# make predictions on the test set
y_predz = logregz.predict(x_test_dr_new)

# confusion matrix
cm = confusion_matrix(y_test_dr_new, y_predz)
print(cm)

# classification Report
print("LBFGS RIDGE RUS Classification Report for Dynamic Rollover: \n",classification_report(y_test_dr_new, y_predz))

# f1 score
f1_z = f1_score(y_test_dr_new, y_predz)
print(f1_z)

# accuracy
accuracyz = accuracy_score(y_test_dr_new, y_predz)
print("Accuracy:", accuracyz)

#####################################################
#####################################################
# PCA Random Undersampling 
#####################################################
#####################################################

from sklearn.decomposition import PCA

pca = PCA(n_components=10)  # Specify the number of principal components to retain
X_train_pca = pca.fit_transform(x_resampled_dr)
X_test_pca = pca.transform(x_test_dr_new)

logreg4 = LogisticRegression()
logreg4.fit(X_train_pca, y_resampled_dr)

y_pred4 = logreg4.predict(X_test_pca)

accuracy_pca10 = accuracy_score(y_test_dr_new, y_pred4)
print("Accuracy:", accuracy_pca10)

f1_4 = f1_score(y_test_dr_new, y_pred4)
print(f1_4)


pca = PCA(n_components=20)  # Specify the number of principal components to retain
X_train_pca = pca.fit_transform(x_resampled_dr)
X_test_pca = pca.transform(x_test_dr_new)

logreg5 = LogisticRegression()
logreg5.fit(X_train_pca, y_resampled_dr)

y_pred5 = logreg5.predict(X_test_pca)

accuracy_pca20 = accuracy_score(y_test_dr_new, y_pred5)
print("Accuracy:", accuracy_pca20)

f1_5 = f1_score(y_test_dr_new, y_pred5)
print(f1_5)


#####################################################
#####################################################
# SAGA RIDGE Random Undersampling
#####################################################
#####################################################

# create and fit the saga logistic regression model
logreg7 = LogisticRegressionCV(Cs=10, cv=5, penalty='l2', solver='saga', random_state=42, max_iter=1000)
logreg7.fit(x_resampled_dr, y_resampled_dr)

# make predictions w/ test set
y_pred7 = logreg7.predict(x_test_dr_new)

# accuracy
accuracy6 = accuracy_score(y_test_dr_new, y_pred7)
print("Accuracy:", accuracy6)

# f1 score
f1_7 = f1_score(y_test_dr_new, y_pred7)
print(f1_7)

#####################################################
#####################################################
# SAGA Basic model Random Undersampling (no penalty)
#####################################################
#####################################################

# create a Logistic Regression model
logreg8 = LogisticRegression(solver='saga', max_iter=1000)

# train the model
logreg8.fit(x_resampled_dr, y_resampled_dr)

# make predictions w/ Test Data
y_pred_imb_dr8 = logreg8.predict(x_test_dr_new)

# accuracy
accuracy5 = accuracy_score(y_test_dr_new, y_pred_imb_dr8)
print("Accuracy:", round(accuracy5,4))

# f1
f1_8 = f1_score(y_test_dr_new, y_pred_imb_dr8)
print(f1_8)

#####################################################
#####################################################
# Logistic Regression Basic Model near miss
#####################################################
#####################################################

# create a Logistic Regression model
logreg9 = LogisticRegression(max_iter=1000)

# train the model
logreg9.fit(x_near_dr, y_near_dr)

# make predictions w/ Test Data
y_pred_imb_dr9 = logreg9.predict(x_test_dr_new)

# accuracy
accuracyn = accuracy_score(y_test_dr_new, y_pred_imb_dr9)
print("Accuracy:", round(accuracyn,4))

# f1 score
f1_9 = f1_score(y_test_dr_new, y_pred_imb_dr9)
print(f1_9)


#####################################################
#####################################################
# LIBLINEAR RIDGE NEAR MISS
#####################################################
#####################################################

# create and fit the  model
logreg11 = LogisticRegressionCV(Cs=10, cv=5, penalty='l2', solver='liblinear', random_state=42, max_iter=1000)
logreg11.fit(x_near_dr, y_near_dr)

# make predictions w/ test set
y_pred11 = logreg11.predict(x_test_dr_new)

# accuracy
accuracy3n = accuracy_score(y_test_dr_new, y_pred11)
print("Accuracy:", accuracy3n)

# f1 score
f1_11 = f1_score(y_test_dr_new, y_pred11)
print(f1_11)

#####################################################
#####################################################
# LBFGS RIDGE NEAR MISS
#####################################################
#####################################################

# create and fit the model
logreg11z = LogisticRegressionCV(Cs=10, cv=5, penalty='l2', solver='lbfgs', random_state=42, max_iter=1000)
logreg11z.fit(x_near_dr, y_near_dr)

# make predictions w/ test set
y_pred11z = logreg11z.predict(x_test_dr_new)

# accuracy
accuracy3nz = accuracy_score(y_test_dr_new, y_pred11z)
print("Accuracy:", accuracy3nz)

# f1 score
f1_11z = f1_score(y_test_dr_new, y_pred11z)
print(f1_11z)

#####################################################
#####################################################
# PCA NEAR MISS
#####################################################
#####################################################

pca = PCA(n_components=10)  # Specify the number of principal components to retain
X_train_pca = pca.fit_transform(x_near_dr)
X_test_pca = pca.transform(x_test_dr_new)

logreg12 = LogisticRegression()
logreg12.fit(X_train_pca, y_near_dr)

y_pred12 = logreg12.predict(X_test_pca)

accuracy_pca10n = accuracy_score(y_test_dr_new, y_pred12)
print("Accuracy:", accuracy_pca10n)

f1_12 = f1_score(y_test_dr_new, y_pred12)
print(f1_12)

pca = PCA(n_components=20)  # Specify the number of principal components to retain
X_train_pca = pca.fit_transform(x_near_dr)
X_test_pca = pca.transform(x_test_dr_new)

logreg13 = LogisticRegression()
logreg13.fit(X_train_pca, y_near_dr)

y_pred13 = logreg13.predict(X_test_pca)

accuracy_pca20n = accuracy_score(y_test_dr_new, y_pred13)
print("Accuracy:", accuracy_pca20n)

f1_13 = f1_score(y_test_dr_new, y_pred13)
print(f1_13)

#####################################################
#####################################################
#SAGA RIDGE NEAR MISS
#####################################################
#####################################################

# create and fit the model
logreg15 = LogisticRegressionCV(Cs=10, cv=5, penalty='l2', solver='saga', random_state=42, max_iter=1000)
logreg15.fit(x_near_dr, y_near_dr)

# make predictions w/ test set
y_pred15 = logreg15.predict(x_test_dr_new)

# accuracy
accuracy6n = accuracy_score(y_test_dr_new, y_pred15)
print("Accuracy:", accuracy6n)

# f1 score
f1_15 = f1_score(y_test_dr_new, y_pred15)
print(f1_15)

#####################################################
#####################################################
# SAGA Basic Model Near Miss Sampling (no penalty)
#####################################################
#####################################################

# create a Logistic Regression model
logreg16 = LogisticRegression(solver='saga', max_iter=1000)

# train the model
logreg16.fit(x_near_dr, y_near_dr)

# make predictions w/ Test Data
y_pred_imb_dr16 = logreg16.predict(x_test_dr_new)

# accuracy
accuracy5n = accuracy_score(y_test_dr_new, y_pred_imb_dr16)
print("Accuracy:", round(accuracy5n,4))

# f1 score
f1_16 = f1_score(y_test_dr_new, y_pred_imb_dr16)
print(f1_16)


############################################################################
############################################################################
# MODEL COMPARISONS
############################################################################
############################################################################

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# round model accuracies
accuracy = round(accuracy,4)
accuracy3 = round(accuracy3,4)#'liblinear RIDGE RUS',
accuracyz= round(accuracyz,4)
accuracy5 = round(accuracy5,4) #'SAGA RUS'
accuracy6 = round(accuracy6,4) #'SAGA RIDGE RUS'
accuracy_pca10 = round(accuracy_pca10,4) #'PCA 10 RUS'
accuracy_pca20 = round(accuracy_pca20,4) #'PCA 20 RUS'
accuracyn = round(accuracyn,4)
accuracy3n = round(accuracy3n,4)
accuracy3nz = round(accuracy3nz,4)
accuracy5n = round(accuracy5n,4)
accuracy6n = round(accuracy6n,4)
accuracy_pca10n = round(accuracy_pca10n,4) 
accuracy_pca20n = round(accuracy_pca20n,4)

# list of model names
model_names = ['Basic Model RUS',  'liblinear RIDGE RUS', 'LBFGS RIDGE RUS', 'PCA 10 RUS', 'PCA 20 RUS','SAGA RUS',  'SAGA RIDGE RUS', 'Basic Model NM',  'liblinear RIDGE NM', 'LBFGS RIDGE NM', 'PCA 10 NM', 'PCA 20 NM','SAGA NM', 'SAGA RIDGE NM'] #, 'SAGA no CV', 'SAGA LASSO CV', 'SAGA RIDGE CV', 'PCA'
accuracies = [accuracy,  accuracy3, accuracyz, accuracy_pca10, accuracy_pca20,accuracy5,  accuracy6, accuracyn,  accuracy3n, accuracy3nz, accuracy_pca10n, accuracy_pca20n, accuracy5n, accuracy6n]

# create bar plot
plt.figure(figsize=(14, 6))
sns.barplot(x=np.arange(len(model_names)), y=accuracies, palette = 'muted')
plt.xticks(np.arange(len(model_names)), model_names)
plt.ylabel('Accuracy')
plt.xlabel('Models')
plt.title('Accuracy Comparison of Logistic Regression Models : Dynamic Rollover')
plt.xticks(rotation=80)
plt.ylim([0, 1])

# add numeric values for each bar
for i, value in enumerate(accuracies):
    plt.annotate(str(value), xy=(i, value), ha='center', va='top')
    
plt.show()

############################################################################
############################################################################
# MODEL COMPARISONS F1
############################################################################
############################################################################

# round f1 values
f1_1 = round(f1_1,4)
f1_3 = round(f1_3,4)
f1_z = round(f1_z,4)
f1_4 = round(f1_4,4)
f1_5 = round(f1_5,4)
f1_7 = round(f1_7,4)
f1_8 = round(f1_8,4)
f1_9 = round(f1_9,4)
f1_11 = round(f1_11,4)
f1_11z = round(f1_11z,4)
f1_12 = round(f1_12,4)
f1_13 = round(f1_13,4)
f1_15 = round(f1_15,4)
f1_16 = round(f1_16,4)

# list of model names
model_names = ['Basic Model RUS',  'liblinear RIDGE RUS', 'LBFGS RIDGE RUS', 'PCA 10 RUS', 'PCA 20 RUS','SAGA RUS',  'SAGA RIDGE RUS', 'Basic Model NM',  'liblinear RIDGE NM', 'LBFGS RIDGE NM', 'PCA 10 NM', 'PCA 20 NM','SAGA NM', 'SAGA RIDGE NM'] #, 'SAGA no CV', 'SAGA LASSO CV', 'SAGA RIDGE CV', 'PCA'
f1_list = [f1_1, f1_3, f1_z, f1_4, f1_5, f1_7, f1_8, f1_9, f1_11, f1_11z, f1_12, f1_13, f1_15, f1_16]

# create bar plot
plt.figure(figsize=(14, 6))
sns.barplot(x=np.arange(len(model_names)), y=f1_list, palette = 'muted')
plt.xticks(np.arange(len(model_names)), model_names)
plt.ylabel('F1 Score')
plt.xlabel('Models')
plt.title('F1 Comparison of Logistic Regression Models : Dynamic Rollover')
plt.xticks(rotation=80)
plt.ylim([0, 1])

# add numeric values for each bar
for i, value in enumerate(f1_list):
    plt.annotate(str(value), xy=(i, value), ha='center', va='top')
    
plt.show()

############################################################################
############################################################################
# VARIABLE IMPORTANCE PLOT For best performing model
############################################################################
############################################################################

# find weights of dependent variables
coefs = logreg3.coef_[0]

# absolute value of the weights
abs_coefs = np.abs(coefs)

# array of variable names
variable_names = np.array(x_train_dr.columns)

# sort coefficients and variable names in descending order of importance
sorted_indices = np.argsort(abs_coefs)[::-1]
sorted_coefs = abs_coefs[sorted_indices]
sorted_variable_names = variable_names[sorted_indices]

# select the top 20 most important variables
top_10_coefs = sorted_coefs[:20]
top_10_variable_names = sorted_variable_names[:20]

# Create a bar plot of variable importance for the top 20 variables
plt.figure(figsize=(10, 6))
sns.barplot(x=np.arange(len(top_10_coefs)), y=top_10_coefs, palette = 'viridis')
plt.xticks(range(len(top_10_coefs)), top_10_variable_names, rotation='vertical')
plt.xlabel('Variables')
plt.ylabel('Absolute Coefficients')
plt.title('Top 20 Variable Importance Plot : Dynamic Rollover Liblinear RIDGE RUS Model')
plt.tight_layout()
plt.show()

############################################################################
############################################################################
# Precision Recall Curve For best performing model
############################################################################
############################################################################

from sklearn.metrics import precision_recall_curve, auc

# calc precision-recall curve
precision, recall, _ = precision_recall_curve(y_test_dr_new, y_predz)

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
# Calibration plot For best performing model
############################################################################
############################################################################


from sklearn.calibration import calibration_curve

# calc calibration curve
prob_true, prob_pred = calibration_curve(y_test_dr_new, y_predz, n_bins=10, strategy='uniform')

# plot the calibration curve
plt.figure(figsize=(8, 6))
plt.plot(prob_pred, prob_true, marker='o', label='Logistic Regression')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly Calibrated')
plt.xlabel('Mean Predicted Probability')
plt.ylabel('Fraction of Positives')
plt.title('Calibration Plot for Dynamic Rollover Logistic Regression')
plt.legend()
plt.show()