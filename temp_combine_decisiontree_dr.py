#!/usr/bin/env python
# coding: utf-8

# In[10]:


import pickle
import os
import re
import glob
import pandas as pd


# In[12]:


os.chdir('C:\RotorCraftData\DAEN690-Karate_Choppers')
print(os.getcwd())


# In[3]:


# Importing Data 

df1 = pd.read_csv("C:\RotorCraftData\CSV1\CSV\SimData_2023.06.15_09.19.14.csv")

## get column names
column_names = list(df1.columns)

## concatenate csv files
## remove first three rows of each file(heading,units, and scenario start rows)
#path = r'C:\RotorCraftData\CSV'
path = r'C:\RotorCraftData\CSV1\CSV'
pattern = r'.*2023\.06\.15.*\.csv$'
folder = glob.glob(os.path.join(path, "*.csv"))
df = (pd.read_csv(f, names = column_names, skiprows=3, skipfooter = 1) for f in folder)
df_concat = pd.concat(df,ignore_index=True)

# Cleaning Data

## remove mostly blank rows
df_concat.dropna(subset=['Elapsed Time'], inplace=True) #run 'count nulls in each column' again to confirm blanks rows were dropped

# Label Data

## Create the new table with date, start time, end time, and maneuver names (based off of Flight Log -- CS handtyped so check for errors!)

label_table = pd.DataFrame({
    'Date': ['2023-06-15', '2023-06-15', '2023-06-15', '2023-06-15', '2023-06-15', '2023-06-15', '2023-06-15', '2023-06-15', '2023-06-15', '2023-06-15', '2023-06-15', '2023-06-15', '2023-06-15', '2023-06-15', '2023-06-15', '2023-06-15', '2023-06-15', '2023-06-15'],  # Replace with actual dates of maneuvers
    'StartTime': ['13:22:15.0', '13:25:25.0', '13:29:25.0', '11:56:25.0', '11:58:03.0', '11:59:51.0', '16:10:04.0', '16:11:41.0', '16:14:20.0', '13:43:12.0', '13:44:10.0', '13:45:19.0', '12:08:11.0', '12:09:31.0', '12:10:51.0', '16:34:28.0', '16:35:06.0', '16:38:26.0'],  # Replace with actual start time of maneuvers
    'EndTime': ['13:22:25.0', '13:25:38.0', '13:29:40.0', '11:56:38.0', '11:58:24.0', '12:00:00.0', '16:10:12.0', '16:11:46.0', '16:14:29.0', '13:43:35.0', '13:44:18.0', '13:45:30.0', '12:08:35.0', '12:09:52.0', '12:11:18.0', '16:34:42.0', '16:35:27.0', '16:38:36.0'],  # Replace with actual end time of maneuvers
    'Label': ['Dynamic Rollover', 'Dynamic Rollover', 'Dynamic Rollover', 'Dynamic Rollover', 'Dynamic Rollover', 'Dynamic Rollover', 'Dynamic Rollover', 'Dynamic Rollover', 'Dynamic Rollover', 'LOW-G', 'LOW-G', 'LOW-G', 'LOW-G', 'LOW-G', 'LOW-G', 'LOW-G', 'LOW-G', 'LOW-G']  # Replace with maneuver names
})

## Convert date, start time, and end time columns to datetime type
label_table['Date'] = pd.to_datetime(label_table['Date'])
label_table['StartTime'] = pd.to_datetime(label_table['StartTime'], format='%H:%M:%S.%f').dt.strftime('%H:%M:%S.%f')
label_table['EndTime'] = pd.to_datetime(label_table['EndTime'], format='%H:%M:%S.%f').dt.strftime('%H:%M:%S.%f')

## Convert the time column on original df to corect format
df_concat['System UTC Time'] = pd.to_datetime(df_concat['System UTC Time'], format='%H:%M:%S.%f').dt.strftime('%H:%M:%S.%f')
## Convert the date column on original df to corect format
df_concat['Date'] = pd.to_datetime(df_concat['Date'])

## Iterate over each row in the new table
for _, row in label_table.iterrows():
    # Extract date, start time, and end time from the current row
    date = row['Date']
    start_time = row['StartTime']
    end_time = row['EndTime']
    label = row['Label']

    ## Filter the existing dataset based on matching date and within start time and end time
    filter_condition = (df_concat['Date'] == date) & (df_concat['System UTC Time'].between(start_time, end_time))
    df_concat.loc[filter_condition, 'Label'] = label

## create dummy variables based on Label

df_dummy = pd.get_dummies(df_concat['Label'], prefix='label')

## convert to int values
df_dummy['label_LOW-G'] = df_dummy['label_LOW-G'].astype(int)
df_dummy['label_Dynamic Rollover'] = df_dummy['label_Dynamic Rollover'].astype(int)

## merge sim data with label df
df = pd.merge(df_concat, df_dummy, how='outer', left_index=True, right_index =True)



## Select Dynamic Rollovever variables

df_dr = df[['Roll Acceleration','Pitch Acceleration','Yaw Acceleration','Roll','Roll Rate','Pitch Rate','Groundspeed','Wind Speed(True)',
      'Wind Direction(Mag)','Gross Weight','Fuel Weight','label_Dynamic Rollover']]

## Select Low G variables

df_lg = df[['Airspeed(True)','Flight Path Angle - VV-[0]','Induced Velo Behind Disc-[0]','Pitch','Pitch Acceleration',
              'Roll','Rotor RPM-[0]','Sideslip Angle','Yaw Acceleration','label_LOW-G']]

with open(f'{path}/dynamic_rollover.pkl', 'wb') as f:
    pickle.dump(df_dr, f)

with open(f'{path}/low_g.pkl', 'wb') as f:
    pickle.dump(df_lg, f)


# In[4]:


df_dr.shape


# In[5]:


df_lg.shape


# In[6]:


cols = df_dr.columns
cols


#  # Decision tree
# 1.  Decision tree train/test split 
# https://www.datacamp.com/tutorial/decision-tree-classification-python

# In[7]:


# Load libraries
import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# In[ ]:


#make a train/test split and scale the predictors with the StandardScaler class:
#X is the predictors, y and z has the response variables.
X = feature_names
y = target1
z = target2
#adding z 
X_train, X_test, y_train, y_test,z_train,z_test = train_test_split(X, y,z, test_size=0.25, random_state=42)

ss = StandardScaler()
#Scaling X 
X_scaled = ss.fit_transform(X)
#Scaling the train and test sets.
X_train_scaled = ss.fit_transform(X_train) # The dataset over which the model will be trained
y_train_scaled = ss.fit_transform(X_train) # This contains the dynamic rollover target
z_train_scaled = ss.fit_transform (X_train) # This contains the LOW-G target
X_test_scaled = ss.transform(X_test)
z_test_scaled = ss.transform(X_test)


# # # 2. Building Decision Tree Model

# In[ ]:


# Create Decision Tree classifer object
DTclf_DRmodel = DecisionTreeClassifier(random_state = 42) # Dynamic Rollover DT model
DTclf_LowGmodel = DecisionTreeClassifier(random_state= 42) # LowG model DT model

# Train Decision Tree Classifer
DTclf_DRmodel = DTclf_DRmodel.fit(X_train, y_train) # DR
DTclf_LowGmodel = DTclf_LowGmodel.fit(X_train, z_train) # LowG

#Predict the response for test dataset
y_pred = DTclf_DRmodel.predict(X_test) # DR
# Predict the probability of each class
y_predProba = DTclf_DRmodel.predict_proba # DR

#Predict the response for test dataset
z_pred = DTclf_LowGmodel.predict(X_test) # LowG
# Predict the probability of each class
z_predProba = DTclf_LowGmodel.predict_proba # LowG


# # # Adding tuning parameters and fitting for Decision tree

# In[ ]:


#Using max_depth, criterion, max_features, min_samples_split
parameters = {'max_depth' : (10,30,50, 70, 90, 100),
             'criterion': ('gine', 'entropy'),
             'max_features' : ('auto', 'sqrt', 'log2'),
             'min_samples_split': (2,4,6, 8, 10)}


# In[ ]:


import warnings
warnings.filterwarnings('ignore')


# In[ ]:


#Dynamic Rollover grid
DT_DRgrid = RandomizedSearchCV(DecisionTreeClassifier(),param_distributions=parameters, cv = 5, verbose = True)
#fit the grid
DT_DRgrid.fit(X_train,y_train)


# In[ ]:


# LowG grid
DT_LowGgrid = RandomizedSearchCV(DecisionTreeClassifier(),param_distributions=parameters, cv = 5, verbose = True)
#fit the grid
DT_LowGgrid.fit(X_train,z_train)


# # # Finding best estimators

# In[ ]:


#Find the best estimator For DR
DT_DRgrid.best_estimator_


# In[ ]:


#Find the best estimator For DR
DT_LowGgrid.best_estimator_


# # # Rebuilding the models based on the best estimator results.

# In[ ]:


#Re build the model with best estimators for DR
DT_DRmodel = DecisionTreeClassifier(criterion='entropy', max_depth=30, max_features='sqrt', 
                                    min_samples_split=6)

DR_model = DT_DRmodel.fit(X_train, y_train)


# In[ ]:


#Re build the model with best estimators
DT_LowGmodel = DecisionTreeClassifier(criterion='entropy',
            max_depth=90, max_features='sqrt',
             min_samples_split=4)
LOWG_model = DT_LowGmodel.fit(X_train, z_train)


# In[ ]:


# imports for classifiers and metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import f1_score
#I'm replacing X_train with X only since I am testing X,Y fitted
#Print accuracy of the DR model
print(f'Train Accuracy - : {DT_DRmodel.score(X_train, y_train): .3f}')
print(f'Train Accuracy - : {DT_DRmodel.score(X_test, y_test): .3f}')
# Print accuracy for LowG model
print(f'Train Accuracy - : {DT_LowGmodel.score(X_train, z_train): .3f}')
print(f'Train Accuracy - : {DT_LowGmodel.score(X_train, z_train): .3f}')


# # # Evaluating the Model

# In[ ]:


# imports for classifiers and metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import f1_score
#Scores without using average setting for f1 score
score_DR_Macro = f1_score(y_pred, y_test)
print('F1 score for Dynamic Rollover:',  score_DR_Macro)
score_LowG = f1_score(z_pred, z_test)
print('F1 score for LowG : ', score_LowG)


# # Dynamic Rollover scores

# In[ ]:


# F1 score
# FORMULA- F1 = 2 * (precision * recall) / (precision + recall)
#'macro':Calculate metrics for each label, and find their unweighted mean. This does not take label imbalance into account.
score_DR_Macro = f1_score(y_pred, y_test, average='macro')
print('Dynamic Rollover f1 Macro score for DR:',score_DR_Macro)
#'micro':Calculate metrics globally by counting the total true positives, false negatives and false positives.
score_DR_Micro = f1_score(y_pred, y_test, average='micro')
print('Dynamic Rollover f1 Micro score for DR:',score_DR_Micro)
# 'weighted':Calculate metrics for each label, and find their average weighted by support (the number of true instances for each label). This alters ‘macro’ to account for label imbalance; it can result in an F-score that is not between precision and recall.
score_DR_W = f1_score(y_pred, y_test, average='weighted')
print('Dynamic Rollover f1 Weighted score for DR:', score_DR_W)


# # Low-G F1 scores

# In[ ]:


score_LowG = f1_score(z_pred, z_test,  average='macro')
print('LowG f1 Macro score for LowG' , score_LowG)
score_LowG_Micro = f1_score(y_pred, z_test, average='micro')
print('LowG f1 Micro score for LowG' , score_LowG_Micro)
score_LowG_W = f1_score(y_pred, z_test, average='weighted')
print('LowG f1 Weighted score for LowG' , score_LowG_W)


# # Confusion matrix and Classification report for Dynamic Rollover

# In[ ]:


from sklearn.model_selection import KFold, cross_val_score 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report




# In[ ]:


#Dynamic Rollover 
cm_DR = confusion_matrix(y_pred, y_test)
print(cm_DR)

#Dynamic Rollover Classification report
print("Classification report for Dynamic Rollover: \n",
      classification_report(y_test, y_pred))


# In[ ]:


#cval_DR = cross_val_score(RCclf_DR, X1, y1, cv=10)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
tn, fp, fn, tp


# In[ ]:


from sklearn.model_selection import KFold, cross_val_score 
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# In[ ]:


#Top left = TrueNegative, Topright= FalseNegative , bottonleft=FalsePositive , bottonright= True positive 
cm = ConfusionMatrixDisplay(confusion_matrix=cm_DR, display_labels=['DR','No DR'])
cm.plot()


# In[ ]:


#Low-G classification report
cm_lg = confusion_matrix(z_pred, z_test)
print(cm_lg)
print("Classification report for LowG: \n",
      classification_report(z_test, z_pred))


# In[ ]:


#Low-G 
 
tn, fp, fn, tp = confusion_matrix(z_test, z_pred).ravel()
tn, fp, fn, tp


# In[ ]:


#Top left = TrueNegative, Topright= FalseNegative , bottonleft=FalsePositive , bottonright= True positive 
cm = ConfusionMatrixDisplay(confusion_matrix=cm_lg, display_labels=['LOW-G','No LOW-G'])
cm.plot()


# # # Visualizing Decision Trees

# In[ ]:


import sklearn
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap, to_rgb
from sklearn.tree import export_graphviz
import graphviz
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier, plot_tree


# In[ ]:


# Retrieve feature names
feature_names = list(X_train.columns)
print(feature_names)


# In[ ]:


#Create a class to store the dependent variable names for labeling the decision tree
# I created it two classes
class_namesdr = '0','1' # DR
class_nameslg = '0', '1'#Low-G


# In[ ]:


# plot decision tree for Dynamic Rollover
plt.figure(figsize=(22, 15))
plot_tree(DT_DRmodel, 
        feature_names= feature_names,
        class_names=class_namesdr , filled=True)
plt.show()


# In[ ]:


# plot LowG tree using the plot_tree function. filled needed to be set to True.
#Creating the classes and features objects helped labeling the tree nodes instead of X#.
plt.figure(figsize=(25, 15))
plot_tree(DT_LowGmodel, 
          feature_names= feature_names,
          class_names=class_nameslg , filled=True)
plt.show()


# In[ ]:


# Visualization of Dynamic Rollover
dot_data = tree.export_graphviz(DT_DRmodel, feature_names= feature_names, class_names=class_namesdr , filled=True) 
DRgraph = graphviz.Source(dot_data, format='png') 
DRgraph.render("DRGraph") 
dot_data = tree.export_graphviz (DT_DRmodel, feature_names= feature_names, class_names=class_namesdr , filled=True,
        rounded = True,
        special_characters=True)
DRgraph=graphviz.Source(dot_data) 
DRgraph


# In[ ]:


# Visualization of LOW-G
dot_LGdata = tree.export_graphviz(DT_LowGmodel, feature_names= feature_names, class_names=class_nameslg , filled=True) 
LowGgraph = graphviz.Source(dot_data, format='png')
LowGgraph.render("LowG-Graph")

dot_LGdata = tree.export_graphviz (DT_LowGmodel, feature_names= feature_names, class_names=class_nameslg , filled=True,
             rounded = True,
        special_characters=True)
LowGgraph=graphviz.Source(dot_LGdata)  
LowGgraph


# # ROC curve

# In[ ]:


#ROC is a plot between Fpositive and true positive rate(tpr).
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve, auc


# In[ ]:


# calculation of precision-recall curve for DT Dynamic Rollover
precision, recall, _ = precision_recall_curve(y_test, y_pred)

# The precision-recall curve for DR
pr_auc = auc(recall, precision)

# plot precision-recall curve
plt.figure(figsize=(8, 8))
plt.plot(recall, precision, color='purple', label=f'Precision Recall AUC = {pr_auc:.2f}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve for Decission Tree Dynamic Rollover ')
plt.legend(loc='lower left')
plt.grid(True)
plt.show()



# In[ ]:


# calculation of precision-recall curve for DT LOW-G
precision, recall, _ = precision_recall_curve(z_test, z_pred)

# The precision-recall curve for DR
pr_auc_lg = auc(recall, precision)

# plot precision-recall curve
plt.figure(figsize=(8, 8))
plt.plot(recall, precision, color='purple', label=f'Precision Recall AUC = {pr_auc:.2f}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve for Decission Tree LOW-G ')
plt.legend(loc='lower left')
plt.grid(True)
plt.show()


# In[ ]:


#ROC curve for Decission Tree Dynamic Rollover 
decisiontree_fpr, decissiontree_tpr, threshold = roc_curve(y_test, y_pred)
auc_decisiontree = auc(decisiontree_fpr,decissiontree_tpr)
plt.figure(figsize=(5,5), dpi=100)
plt.plot(decisiontree_fpr,decissiontree_tpr, marker='.', label='Decission Tree (auc = %0.3f)'% auc_decisiontree)
plt.title('ROC curve for Decission Tree Dynamic Rollover ')
plt.xlabel('False Positive Rate ')
plt.ylabel('True Positive Rate ')
plt.legend()

plt.show()


# In[ ]:


#ROC for LOW_G
decisiontree_fpr, decissiontree_tpr, threshold = roc_curve(z_test, z_pred)
auc_decisiontree = auc(decisiontree_fpr,decissiontree_tpr)
plt.figure(figsize=(5,5), dpi=100)
plt.plot(decisiontree_fpr,decissiontree_tpr, marker='.', label='Decission Tree LOW-G (auc = %0.3f)'% auc_decisiontree)
plt.title('ROC curve for Decission Tree LOW-G ')
plt.xlabel('False Positive Rate ')
plt.ylabel('True Positive Rate ')
plt.legend()

plt.show()

