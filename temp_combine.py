# -*- coding: utf-8 -*-
"""
Created on Sat Jul 22 15:27:31 2023

@author: jetow
"""

import pandas as pd
import glob
import os
import pickle
# Importing Data 

df1 = pd.read_csv("../CSV/SimData_2023.06.15_09.19.14.csv")

## get column names
column_names = list(df1.columns)

## concatenate csv files
## remove first three rows of each file(heading,units, and scenario start rows)
path = r'../CSV'
folder = glob.glob(os.path.join(path, "*.csv"))
df = (pd.read_csv(f, names = column_names, skiprows=3, skipfooter = 1, engine='python') for f in folder)
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
