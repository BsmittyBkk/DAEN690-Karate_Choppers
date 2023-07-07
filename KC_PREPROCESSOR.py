# This file will serve as a consolidated pre-processing script for the
# team to all work off of a cleaned master dataframe

########################################################################

# The first thing will be to import the libraries used

import pandas as pd
import numpy as np
import glob
import os

# The next step involves having all of the working CSV's within the
# current directory

# First read in a CSV to get the column names and then delete it from
# working space. The name of the file could change depending on your
# working directory.

df1 = pd.read_csv("SimData_2023.06.13_09.48.56.csv")

# Next grab the column names which will be appended to the concatenated DF

column_names = list(df1.columns)

# Clear working space

del df1

# Next all CSV's in the current directory will be merged together.
# The 'path' variable will be unique to every developer. This variable
# should be populated with the directory where the CSV's are located

path = r'/Volumes/Backup Plus/Brett DAEN 690/6:15:23-Flight-Data'
folder = glob.glob(os.path.join(path, "*.csv"))
df = (pd.read_csv(f, names = column_names, skiprows=3, skipfooter=1) for f in folder)
df_concat = pd.concat(df, ignore_index=True)

# Next check the size of the combined dataframe
print(df_concat.shape)

# Check some basic statistical information about the dataframe

print(df_concat.describe())

# Check for the null values by column. We know that in the 'Elapsed Time' column, there are nulls due to entries
# stating when a manuever ends or begins, we will drop these instances

print(df_concat.isna().sum())

# Drop the NaN entries due to the simulator status
df_concat.dropna(subset=['Elapsed Time'], inplace=True)

# Check for columns where blanks remain
# Check for columns where infinite values remain

number_inf = df_concat[df_concat == np.inf].count()

print('Remaining columns with null values')
print('-------------------------------------')
print(df_concat.columns[df_concat.isna().sum() > 0])

print('Remaining columns with infinite values')
print('-------------------------------------')
print(number_inf[number_inf > 0])

# Before dropping the NaN columns, we consulted with our clients and they agreed there would not be significant information loss
# by dropping these. First we will convert the infinite values into NaN's so that all can be dropped in 1 go.

df_concat.replace([np.inf, -np.inf], np.nan, inplace=True)

# Gather all NaN columns

col_nan = [col for col in df_concat.columns if df_concat[col].isna().sum() > 0]
print(col_nan)
df_concat.drop(col_nan, axis=1, inplace=True)

# Next identify columns with only 1 unique value and drop these. These columns are near zero or zero variance and do not add
# to modeling power

columns_with_one_unique_value = []
for column in df_concat.columns:
    if df_concat[column].nunique() == 1:
        columns_with_one_unique_value.append(column)

print(columns_with_one_unique_value)
df_concat.drop(columns_with_one_unique_value, axis=1, inplace=True)

# Next, the data will be labeled using the information from the flight logs

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

# Convert the time column on original df to correct format
df_concat['System UTC Time'] = pd.to_datetime(df_concat['System UTC Time'], format='%H:%M:%S.%f').dt.strftime('%H:%M:%S.%f')
# Convert the date column on original df to correct format
df_concat['Date'] = pd.to_datetime(df_concat['Date'])

# Apply the labeling to the dataset

for _, row in label_table.iterrows():
    # Extract date, start time, and end time from the current row
    date = row['Date']
    start_time = row['StartTime']
    end_time = row['EndTime']
    label = row['Label']

    # Filter the existing dataset based on matching date and within start time and end time
    filter_condition = (df_concat['Date'] == date) & (df_concat['System UTC Time'].between(start_time, end_time))
    df_concat.loc[filter_condition, 'Label'] = label

# Set the NaN values in Label Row to 'Other'

df_concat['Label'] = df_concat['Label'].fillna('Other')
df_pre_proc = df_concat

# df_pre_proc can be used as a dataframe to work out of where the Labels have been applied, all nulls and infinites have been
# dropped and all zero variance variables have been dropped.

# Below is a dataframe where all 'object' type variables outside of the Label variable have been dropped
# First pull out the 'Label' column because this is an object type

df_label = df_pre_proc['Label']
df_no_obj = df_pre_proc.select_dtypes(exclude=['object'])
df_no_obj = df_no_obj.join(df_label)

del df_label

# df_no_obj can be used as a dataframe without any categorical variables. It should be noted that there is a variable with
# 'datetime64[ns]' remaining. This could cause problems, if you would like to drop this variable use the code below.

df_no_obj = df_no_obj.select_dtypes(exclude=['datetime64[ns]'])

# This same line may be used on the df_pre_proc dataframe.
