# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 22:36:37 2023

@author: 18043
"""

import os
import pandas as pd
import glob


#load in 1 file to get the column names
df1 = pd.read_csv('D:/CSV/SimData_2023.06.13_09.48.56.csv')

#get column names
column_names = list(df1.columns)
print(column_names)

#delete df1 from above to save space
del df1

#Jane's code for loading CSV's into one DF
path = r'D:\CSV'
folder = glob.glob(os.path.join(path, "*.csv"))
df = (pd.read_csv(f, names = column_names, skiprows = 3, skipfooter=1) for f in folder)
df_comb = pd.concat(df,ignore_index=True)

#check first and last rows to make sure files loaded into df correctly
print(df_comb.head())
print(df_comb.tail())

#OPTIONAL: change print out of rows and columns to not show "..."
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

#Shape of the data set (number of rows, number of columns)
print(df_comb.shape)

#Basic statistical information
print(df_comb.describe())

#Data types
print(df_comb.dtypes)

#Count missing values in each column
print(df_comb.isnull().sum())

#Count unique values in each column
print(df_comb.nunique())

#Columns with only 1 distinct value (noise in the dataset? potentially drop?)
columns_with_one_unique_value = []
for column in df_comb.columns:
    if df_comb[column].nunique() == 1:
        columns_with_one_unique_value.append(column)
print(columns_with_one_unique_value)

# Print the list from above with each column on a new line (CS added to PPT)
print("Columns with only 1 unique value:")
for column in columns_with_one_unique_value:
    print(column)

# plot histograms of numeric columns:
numeric_cols = df_comb.select_dtypes(include=['int', 'float']).columns
df_comb[numeric_cols].hist(bins=10, figsize=(30, 35))

print(numeric_cols)

# Calc Variance for each col
variance = df_comb[numeric_cols].var()

# Setting a threshold for variance -- thoughts?
threshold = 0.01

# Cols with near zero variance
near_zero_variance_columns = variance[variance <= threshold].index.tolist()
print(near_zero_variance_columns)

####################################
#CS Attempt at "automated" labeling of dataset
####################################

# Create the new table with date, start time, end time, and maneuver names (based off of Flight Log -- CS handtyped so check for errors!)
label_table = pd.DataFrame({
    'Date': ['2023-06-15', '2023-06-15', '2023-06-15', '2023-06-15', '2023-06-15', '2023-06-15', '2023-06-15', '2023-06-15', '2023-06-15', '2023-06-15', '2023-06-15', '2023-06-15', '2023-06-15', '2023-06-15', '2023-06-15', '2023-06-15', '2023-06-15', '2023-06-15'],  # Replace with actual dates of maneuvers
    'StartTime': ['13:22:15.0', '13:25:25.0', '13:29:25.0', '11:56:25.0', '11:58:03.0', '11:59:51.0', '16:10:04.0', '16:11:41.0', '16:14:20.0', '13:43:12.0', '13:44:10.0', '13:45:19.0', '12:08:11.0', '12:09:31.0', '12:10:51.0', '16:34:28.0', '16:35:06.0', '16:38:26.0'],  # Replace with actual start time of maneuvers
    'EndTime': ['13:22:25.0', '13:25:38.0', '13:29:40.0', '11:56:38.0', '11:58:24.0', '12:00:00.0', '16:10:12.0', '16:11:46.0', '16:14:29.0', '13:43:35.0', '13:44:18.0', '13:45:30.0', '12:08:35.0', '12:09:52.0', '12:11:18.0', '16:34:42.0', '16:35:27.0', '16:38:36.0'],  # Replace with actual end time of maneuvers
    'Label': ['Dynamic Rollover', 'Dynamic Rollover', 'Dynamic Rollover', 'Dynamic Rollover', 'Dynamic Rollover', 'Dynamic Rollover', 'Dynamic Rollover', 'Dynamic Rollover', 'Dynamic Rollover', 'LOW-G', 'LOW-G', 'LOW-G', 'LOW-G', 'LOW-G', 'LOW-G', 'LOW-G', 'LOW-G', 'LOW-G']  # Replace with maneuver names
})

# Convert date, start time, and end time columns to datetime type
label_table['Date'] = pd.to_datetime(label_table['Date'])
label_table['StartTime'] = pd.to_datetime(label_table['StartTime'], format='%H:%M:%S.%f').dt.strftime('%H:%M:%S.%f')
label_table['EndTime'] = pd.to_datetime(label_table['EndTime'], format='%H:%M:%S.%f').dt.strftime('%H:%M:%S.%f')

# Convert the time column on original df to corect format
df_comb['System UTC Time'] = pd.to_datetime(df_comb['System UTC Time'], format='%H:%M:%S.%f').dt.strftime('%H:%M:%S.%f')
# Convert the date column on original df to corect format
df_comb['Date'] = pd.to_datetime(df_comb['Date'])


# Iterate over each row in the new table
for _, row in label_table.iterrows():
    # Extract date, start time, and end time from the current row
    date = row['Date']
    start_time = row['StartTime']
    end_time = row['EndTime']
    label = row['Label']

    # Filter the existing dataset based on matching date and within start time and end time
    filter_condition = (df_comb['Date'] == date) & (df_comb['System UTC Time'].between(start_time, end_time))
    df_comb.loc[filter_condition, 'Label'] = label

# Print the updated dataset
print(df_comb)
#check that any of the Label column is populated by printing the records where that column is not null
print(df_comb[df_comb['Label'].notnull()])

#count of each distinct value in new Label column
value_counts = df_comb['Label'].value_counts()
print(value_counts)

