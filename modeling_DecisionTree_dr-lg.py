#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pickle
import os
import re
import glob
import pandas as pd


# In[2]:


os.chdir('C:\RotorCraftData\DAEN690-Karate_Choppers')
print(os.getcwd())


# In[3]:


# this is the path to the folder where you have the CSVs, NO OTHER CSVs SHOULD BE PRESENT
# please make sure this path is not inside the scope of GitHub so we do not go over on data for our repo
path = r'C:\RotorCraftData\CSV'
pattern = r'.*2023\.06\.15.*\.csv$'

# this imports a list of columns that was saved after the removal of variance on a single CSV, this list will be used to define which columns to read in
with open('src./use_cols_aws.pkl', 'rb') as f: 
    use_cols = pickle.load(f)

# the data will be labeled using the information from the flight logs
label_table = pd.DataFrame({
    'Date': ['2023-06-15', '2023-06-15', '2023-06-15', '2023-06-15', '2023-06-15', '2023-06-15', '2023-06-15', '2023-06-15', '2023-06-15', '2023-06-15', '2023-06-15', '2023-06-15', '2023-06-15', '2023-06-15', '2023-06-15', '2023-06-15', '2023-06-15', '2023-06-15'], 
    # Replace with actual start time of maneuvers
    'StartTime': ['13:22:15.0', '13:25:25.0', '13:29:25.0', '11:56:25.0', '11:58:03.0', '11:59:51.0', '16:10:04.0', '16:11:41.0', '16:14:20.0', '13:43:12.0', '13:44:10.0', '13:45:19.0', '12:08:11.0', '12:09:31.0', '12:10:51.0', '16:34:28.0', '16:35:06.0', '16:38:26.0'],
    # Replace with actual end time of maneuvers
    'EndTime': ['13:22:25.0', '13:25:38.0', '13:29:40.0', '11:56:38.0', '11:58:24.0', '12:00:00.0', '16:10:12.0', '16:11:46.0', '16:14:29.0', '13:43:35.0', '13:44:18.0', '13:45:30.0', '12:08:35.0', '12:09:52.0', '12:11:18.0', '16:34:42.0', '16:35:27.0', '16:38:36.0'],
    'Label': ['Dynamic Rollover', 'Dynamic Rollover', 'Dynamic Rollover', 'Dynamic Rollover', 'Dynamic Rollover', 'Dynamic Rollover', 'Dynamic Rollover', 'Dynamic Rollover', 'Dynamic Rollover', 'LOW-G', 'LOW-G', 'LOW-G', 'LOW-G', 'LOW-G', 'LOW-G', 'LOW-G', 'LOW-G', 'LOW-G']  
})

# convert date, start time, and end time columns to datetime type
label_table['Date'] = pd.to_datetime(label_table['Date'])
label_table['StartTime'] = pd.to_datetime(
    label_table['StartTime'], format='%H:%M:%S.%f').dt.strftime('%H:%M:%S.%f')
label_table['EndTime'] = pd.to_datetime(
    label_table['EndTime'], format='%H:%M:%S.%f').dt.strftime('%H:%M:%S.%f')


def combine_csv_files(csv_directory, columns_to_use, label_df):
    # get list of CSV file paths in the directory
    csv_files = [os.path.join(csv_directory, filename) for filename in os.listdir(
        csv_directory) if re.match(pattern, filename)]
    # create an empty dataframe to store the combined data
    combined_df = pd.DataFrame()

    # iterate over each CSV file
    for file in csv_files:
        # read CSV file and select desired columns
        temp_df = pd.read_csv(file, usecols=columns_to_use, names=columns_to_use, skiprows=3, skipfooter=1, engine='python')
        # drop rows that Elapsed Time are mostly null, these are the breaks in simulation
        temp_df.dropna(subset=['Elapsed Time'], inplace=True)
        # temp_df.drop(['Elapsed Time'], inplace=True)
        temp_df.dropna(inplace=True)
        # concatenate the temporary dataframe with the running dataframe
        combined_df = pd.concat([combined_df, temp_df], ignore_index=True)

    # convert the time column on original df to correct format
    combined_df['System UTC Time'] = pd.to_datetime(
    combined_df['System UTC Time'], format='%H:%M:%S.%f').dt.strftime('%H:%M:%S.%f')
    # convert the date column on original df to correct format
    combined_df['Date'] = pd.to_datetime(combined_df['Date'])
    
    # apply the labeling to the dataset
    for _, row in label_df.iterrows():
        # extract date, start time, and end time from the current row
        date = row['Date']
        start_time = row['StartTime']
        end_time = row['EndTime']
        label = row['Label']

        # filter the existing dataset based on matching date and within start time and end time
        filter_condition = (combined_df['Date'] == date) & (
            combined_df['System UTC Time'].between(start_time, end_time))
        combined_df.loc[filter_condition, 'Label'] = label
    dummies_df = pd.get_dummies(combined_df['Label'], dummy_na=False)
    dummies_df = dummies_df.astype(int)
    combined_df = pd.concat([combined_df, dummies_df], axis=1)
    # Convert the time column to pandas datetime format if it's not already in that format
    combined_df['System UTC Time'] = pd.to_datetime(combined_df['System UTC Time'], format='%H:%M:%S.%f')

    # Set the start and end time range
    start_time = pd.to_datetime('11:56:25.0', format='%H:%M:%S.%f')
    end_time = pd.to_datetime('16:38:26.0', format='%H:%M:%S.%f')

    # Filter the DataFrame to include rows between the start and end times
    combined_df = combined_df[(combined_df['System UTC Time'] >= start_time) & (combined_df['System UTC Time'] <= end_time)].copy()

    combined_df.drop(['Elapsed Time', 'Date', 'System UTC Time', 'Label'], inplace=True, axis=1)
    
    return combined_df

# this calls the function from above that cleans and creates dummy variables for our target variables
df = combine_csv_files(path, use_cols, label_table)
# this will create a pickle file with the working dataframe in your directory with the original CSV files
# you will not need to run this script again, as we will load in the dataframe from the pickle file
with open(f'{path}/working_df_aws.pkl', 'wb') as f:
    pickle.dump(df, f)


# In[4]:


#Based on the df corr matrix, we can see that there are two cols with zero varianec. I will remove them. 
df2 = df.drop(['GPS 1 DME Time', 'NAV 2 DME Time'], axis = 1)


#  # Decision tree
# 1.  Decision tree train/test split 
# https://www.datacamp.com/tutorial/decision-tree-classification-python

# In[5]:


# Load libraries
import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# In[6]:


#Create feature_names and target varaibles
feature_names = pd.DataFrame(df, columns=['Heading(mag)', 'Baro Setting Pilot', 'Ground Track Copilot',
       'Yaw Rate', 'Turn Rate', 'Flight Path Angle - VV-[0]',
       'Flight Path Angle - VV-[1]', 'Flight Path Angle - VV-[2]',
       'Ground Track - VV-[2]', 'Yaw Acceleration', 'Acceleration in Latitude',
       'Acceleration in Normal', 'Right Brake Pos', 'TOGA Status',
       'AP1 Status', 'GPS 1 NAV ID', 'NAV 2 NAV ID', 'NAV 2 DME Distance',
       'NAV 2 DME Speed', 'FMS Waypoints', 'Nav1 Ver Deviation',
       'Tail Rotor Chip Warning', 'Transmission Chip Warning',
       'Transmission Oil Temp Warning'])
feature_names.shape


# In[7]:


target1 = df['Dynamic Rollover']
target1.head()
print(target1.isnull().sum())


# In[8]:


target2 = df['LOW-G']
target2.head()
print(target2.isnull().sum())


# In[9]:


#make a train/test split and scale the predictors with the StandardScaler class:
#X is the predictors, y and z has the response variables.
X = feature_names
y = target1
z = target2
#adding z to test a 3d plot
X_train, X_test, y_train, y_test,z_train,z_test = train_test_split(X, y,z, test_size=0.25, random_state=42)

ss = StandardScaler()
#Scaling X 
X_scaled = ss.fit_transform(X)
#Scaling the train and test sets.
X_train_scaled = ss.fit_transform(X_train)
X_test_scaled = ss.transform(X_test)


# # # 2. Building Decision Tree Model

# In[10]:


# Create Decision Tree classifer object
DTclf_DRmodel = DecisionTreeClassifier() # Dynamic Rollover DT model
DTclf_LowGmodel = DecisionTreeClassifier() # LowG model DT model

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

# In[11]:


#Using max_depth, criterion, max_features, min_samples_split
parameters = {'max_depth' : (10,30,50, 70, 90, 100),
             'criterion': ('gine', 'entropy'),
             'max_features' : ('auto', 'sqrt', 'log2'),
             'min_samples_split': (2,4,6, 8, 10)}


# In[12]:


import warnings
warnings.filterwarnings('ignore')


# In[13]:


#Dynamic Rollover grid
DT_DRgrid = RandomizedSearchCV(DecisionTreeClassifier(),param_distributions=parameters, cv = 5, verbose = True)
#fit the grid
DT_DRgrid.fit(X_train,y_train)


# In[14]:


# LowG grid
DT_LowGgrid = RandomizedSearchCV(DecisionTreeClassifier(),param_distributions=parameters, cv = 5, verbose = True)
#fit the grid
DT_LowGgrid.fit(X_train,z_train)


# # # Finding best estimators

# In[15]:


#Find the best estimator For DR
DT_DRgrid.best_estimator_


# In[16]:


#Find the best estimator For DR
DT_LowGgrid.best_estimator_


# # # Rebuilding the models based on the best estimator results.

# In[17]:


#Re build the model with best estimators for DR
DT_DRmodel = DecisionTreeClassifier(criterion='entropy', max_depth=100, max_features='sqrt', 
                                    min_samples_split=10)

DT_DRmodel.fit(X_train, y_train)


# In[18]:


#Re build the model with best estimators
DT_LowGmodel = DecisionTreeClassifier(criterion='entropy', max_depth=50, max_features='log2',
                       min_samples_split=10)
DT_LowGmodel.fit(X_train, z_train)


# In[19]:


# imports for classifiers and metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import f1_score

#Print accuracy of the DR model
print(f'Train Accuracy - : {DT_DRmodel.score(X_train, y_train): .3f}')
print(f'Train Accuracy - : {DT_DRmodel.score(X_test, y_test): .3f}')
# Print accuracy for LowG model
print(f'Train Accuracy - : {DT_LowGmodel.score(X_train, z_train): .3f}')
print(f'Train Accuracy - : {DT_LowGmodel.score(X_train, z_train): .3f}')


# # # Evaluating the Model

# In[20]:


# imports for classifiers and metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import f1_score
#Scores without using average setting for f1 score
score_DR_Macro = f1_score(y_pred, y_test)
print('F1 score for Dynamic Rollover:',  score_DR_Macro)
score_LowG = f1_score(z_pred, z_test)
print('F1 score for LowG : ', score_LowG)


# # Dynamic Rollover scores

# In[21]:


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

# In[22]:


score_LowG = f1_score(z_pred, z_test,  average='macro')
print('LowG f1 Macro score for LowG' , score_LowG)
score_LowG_Micro = f1_score(y_pred, z_test, average='micro')
print('LowG f1 Micro score for LowG' , score_LowG_Micro)
score_LowG_W = f1_score(y_pred, z_test, average='weighted')
print('LowG f1 Weighted score for LowG' , score_LowG_W)


# #Confusion matrix

# In[23]:


from sklearn.model_selection import KFold, cross_val_score 
from sklearn.metrics import confusion_matrix

cm_DR = confusion_matrix(y_pred, y_test, normalize = 'all')
print(cm_DR)
#cval_DR = cross_val_score(RCclf_DR, X1, y1, cv=10)
 
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
tn, fp, fn, tp


# #Dynamic Rollover Classification report

# In[24]:


from sklearn.metrics import classification_report

#Dynamic Rollover Classification report
print("Classification report for Dynamic Rollover: \n",classification_report(y_test, y_pred))
             


# #Low-G classification report

# In[25]:


print("Classification report for LowG: \n",classification_report(z_test, z_pred))


# # # Visualizing Decision Trees

# In[26]:


import sklearn
from sklearn.tree import export_graphviz
import graphviz
from sklearn import tree


# In[27]:


#Plot Dynamic Rollover tree
tree.plot_tree(DT_DRmodel)


# In[28]:


# plot LowG tree using the plot_tree function
tree.plot_tree(DT_LowGmodel)


# In[29]:


# Visualization of Dynamic Rollover
dot_data = tree.export_graphviz(DT_DRmodel, out_file=None) 
DRgraph = graphviz.Source(dot_data) 
DRgraph.render("DRGraph") 

dot_data = tree.export_graphviz (DT_DRmodel, out_file = None,
              filled=True, 
        rounded = True,
        special_characters=True)
DRgraph=graphviz.Source(dot_data)  
DRgraph


# In[30]:


# Visualization of LOW-G
dot_LGdata = tree.export_graphviz(DT_LowGmodel, out_file=None) 
LowGgraph = graphviz.Source(dot_LGdata) 
LowGgraph.render("LowG-Graph")

dot_LGdata = tree.export_graphviz (DT_LowGmodel, out_file = None,
              filled=True, 
        rounded = True,
        special_characters=True)
LowGgraph=graphviz.Source(dot_LGdata)  
LowGgraph


# # # Creating pipelines

# In[31]:


import imblearn
from collections import Counter
from sklearn.datasets import make_classification
from imblearn.over_sampling import RandomOverSampler
from numpy import mean
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
import warnings
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier


# In[32]:


# # create pipeline, it will include the following after dividing the data into train/test splits:- from video name ML Pipeline
# 1. Data preprocessing using MinMax Scaler
# 2. Reducing Dimensionality using PCA
# 3. Training respective models


# In[33]:


feature_names.shape


# In[34]:


#make a train/test split and scale the predictors with the StandardScaler class:
#X is the predictors, y and z has the response variables.
X = feature_names # predictors
y = target1 # response Var
z = target2
#adding z to test a 3d plot
X_train, X_test, y_train, y_test,z_train,z_test = train_test_split(X, y,z, test_size=0.25, random_state=42)

ss = StandardScaler()
#Scaling X 
X_scaled = ss.fit_transform(X)
#Scaling the train and test sets.
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)


# In[35]:


#Decision tree Pipeline
DecisionTreePipeline = Pipeline([('myscaler', MinMaxScaler()),
                                ('mypca', PCA(n_components=3)),
                                ('decisiontree_classifier',DecisionTreeClassifier())])


# In[36]:


#Random Forest Pipeline
RandomForestPipeline = Pipeline([('myscaler', MinMaxScaler()),
                                ('mypca', PCA(n_components=3)),
                                ('randomforest_classifier',RandomForestClassifier())])


# In[37]:


#Logistic regression Pipeline
LogisticRegressionPipeline = Pipeline([('myscaler', MinMaxScaler()),
                                ('mypca', PCA(n_components=3)),
                              ('logistic_classifier',LogisticRegression())])


# In[38]:


# SVM Pipeline
SvmPipeline = Pipeline([('myscaler', MinMaxScaler()),
                                ('mypca', PCA(n_components=3)),
                                ('Svm',SVC())])


# In[39]:


# KNN Pipeline
KnnPipeline = Pipeline([('myscaler', MinMaxScaler()),
                                ('mypca', PCA(n_components=3)),
                                ('KNN',KNeighborsClassifier())])


# # # Model training and validation

# In[40]:


#Difining the pipelines in a list
mypipeline = [DecisionTreePipeline,RandomForestPipeline,LogisticRegressionPipeline,SvmPipeline, KnnPipeline]


# In[41]:


#Difine variables for Choosing best model
accuracy = 0.0
classifier= 0
pipeline = ""


# In[42]:


#Creating dictionary of pipelines and training models
PipelineDict = {0:'Decision Tree', 1: 'Random Forest', 2: 'Logistic Regression', 3: 'Svm', 4: 'KNN'}


# In[43]:


warnings.filterwarnings('ignore')


# In[44]:


#Fit the pipelines
for mypipe in mypipeline: 
    mypipe.fit(X_train, y_train)


# In[45]:


#getting test accuracy for all classifiers
for i, model in enumerate(mypipeline):
    print("{}Test Accuracy: {}" .format(PipelineDict[i], model.score(X_test, y_test)))


# In[46]:


#Choosing best model fro the given data
for i, model in enumerate(mypipeline):
    if model.score(X_test, y_test)> accuracy:
        accuracy=model.score(X_test,y_test)
        pipeline = model
        classifier = i
print('Classifier with best accurarcy:{}' .format(PipelineDict[classifier]))

