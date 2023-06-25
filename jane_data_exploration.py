# -*- coding: utf-8 -*-
"""
Created on Sun Jun 18 14:04:30 2023

@author: jetow
"""

import pandas as pd
import glob
import os
# import matplotlib.pyplot as plt
# import seaborn as sns
# import matplotlib.pyplot as plt


# Importing Data 

df1 = pd.read_csv("Data/SimData_2023.06.13_09.48.56.csv")


## get units
units = df1.iloc[0]

## get column names
column_names = list(df1.columns)

## concatenate csv files
## remove first three rows of each file(heading,units, and scenario start rows)
path = r'C:\DAEN690\GitHub\DAEN690-Karate_Choppers\Data'
folder = glob.glob(os.path.join(path, "*.csv"))
df = (pd.read_csv(f, names = column_names, skiprows=3, skipfooter = 1) for f in folder)
df_concat = pd.concat(df,ignore_index=True)

# Get basic dataset info
len(df_concat)
df_concat.shape
df_concat.head()

pd.set_option("display.max.columns", None)
df_concat.tail()

df_concat.info()

df_concat.describe()
df_concat.describe(include=object)

# Cleaning Data


## find rows with missing data
rows_without_missing_data = df_concat.dropna()
rows_without_missing_data.shape #all rows have missing data

## count nulls in each column
null_values = df_concat.isnull().sum()
null_values.sort_values(ascending=False) 
# NAV 2 NAV ID is completely blank (591091)
# NAV 1 NAV ID has 59% blank rows (346284)
# Transmission Oil Pressure Warning has <1% blank rows (4177)

nan_in_col = df_concat[df_concat['Elapsed Time'].isnull()]
nan_null_col = nan_in_col.isnull().sum()
nan_null_col.sort_values(ascending=False) 
print(nan_in_col)  # 13 rows are mostly blank

## check non-null values is mostly empty rows
nan_in_col_val = nan_in_col[nan_in_col.columns[~nan_in_col.isnull().all()]]
nan_in_col_val #these records are where scenarios were paused and resumed

## remove mostly blank rows
df_concat.dropna(subset=['Elapsed Time'], inplace=True) #run 'count nulls in each column' again to confirm blanks rows were dropped


# Check for Duplicates
df_duplicates = df_concat[df_concat.duplicated()] #no duplicates found


# Check Column Names and Data Types
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(df_concat.columns)

data_types = df_concat.dtypes
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(data_types)


# Get Numeric Columns
no_columns = len(df_concat.columns)
columns = df_concat.columns
columns = list(columns)

df_concat.select_dtypes('float64','int64')
num_columns = list(df_concat.select_dtypes('float64','int64'))
num_columns

# Plot Distributions

# sns.set(style="darkgrid")

# for i in num_columns:
#     plt.figure()
#     plt.tight_layout()
#     sns.set(rc={"figure.figsize":(8,5)})
    
#     f, (ax_box, ax_hist) = plt.subplots(2, sharex=True)
#     plt.gca().set(xlabel= i,ylabel='Frequency')
#     sns.boxplot(df_concat[i], ax=ax_box , linewidth= 1.0)
#     sns.histplot(df_concat[i], ax=ax_hist , bins = 10,kde=True)


# Variance

variance = df_concat[num_columns].var()

## Near Zero Variance
variance = df_concat[num_columns].var()
nzv_columns = variance[variance <= .001].index.tolist()
print(nzv_columns)
len(nzv_columns) #37

## drop near zero variance columns

for column in nzv_columns:
    df_concat.drop([column], inplace=True, axis = 1)


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

## Print the updated dataset
print(df_concat)
## check that any of the Label column is populated by printing the records where that column is not null
print(df_concat[df_concat['Label'].notnull()])

## count of each distinct value in new Label column
value_counts = df_concat['Label'].value_counts()
print(value_counts)


# Label Covariance

with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(df_concat.dtypes)

## check object fields

df_concat.select_dtypes('object')
obj_columns = list(df_concat.select_dtypes('object'))
obj_columns 

# ['System UTC Time',
#  'Latitude',
#  'Autopilot Modes',
#  'Flight ID',
#  'GPS 1 NAV ID',
#  'NAV 1 NAV ID',
#  'NAV 2 NAV ID',
#  'FMS Waypoints',
#  'Actual Visibility',
#  'Cloud Type 0',
#  'Cloud Type 1',
#  'Cloud Type 2',
#  'Label']

## checked all object variables manually
pd.crosstab(index=df_concat['Label'], columns=df_concat['Cloud Type 2'])
#none of them seem like they will be helpful (no covariance)

## ANOVA

from scipy.stats import f_oneway

df_concat.select_dtypes('float64','int64')
num_columns = list(df_concat.select_dtypes('float64','int64'))
num_columns

p_values = {}

for column in num_columns:
    group_list = df_concat.groupby('Label')[column].apply(list)
    res_anova = f_oneway(*group_list)
    if res_anova[1] <= 0.05:
        p_values[column] = (res_anova[1])
    
print(p_values)  
len(p_values) 
sig_cols = list(p_values.keys())       
print(sig_cols)    
len(sig_cols)  # 97 potentially significant fields    

# Low G

## create dummy variables based on Label

df_dummy = pd.get_dummies(df_concat['Label'], prefix='label')

df_dummy.dtypes

## convert to int values
# df_dummy['label_LOW-G'] = df_dummy['label_LOW-G'].astype(int)
# df_dummy['label_Dynamic Rollover'] = df_dummy['label_Dynamic Rollover'].astype(int)

lowg_counts = df_dummy['label_LOW-G'].value_counts()
print(lowg_counts)
dr_counts = df_dummy['label_Dynamic Rollover'].value_counts()
print(dr_counts)

df_concat_dummy = pd.merge(df_concat, df_dummy, how='outer', left_index=True, right_index =True)

## Low G ANOVA   

lowg_p_values = {}

for column in num_columns:
    group_list = df_concat_dummy.groupby('label_LOW-G')[column].apply(list)
    lowg_anova = f_oneway(*group_list)
    if lowg_anova[1] <= 0.05:
        lowg_p_values[column] = (lowg_anova[1])
    
print(lowg_p_values)  
len(lowg_p_values) 
lowg_sig_cols = list(lowg_p_values.keys())       
print(lowg_sig_cols)    
len(lowg_sig_cols)  # 133 potentially significant fields 


## Dynamic Rollover ANOVA 
  
dr_p_values = {}

for column in num_columns:
    group_list = df_concat_dummy.groupby('label_Dynamic Rollover')[column].apply(list)
    dr_anova = f_oneway(*group_list)
    if dr_anova[1] <= 0.05:
        dr_p_values[column] = (dr_anova[1])
    
print(dr_p_values)  
len(dr_p_values) 
dr_sig_cols = list(dr_p_values.keys())       
print(dr_sig_cols)    
len(dr_sig_cols) # 139 potentially significant fields

    
    
    
    
    
    
    
    
