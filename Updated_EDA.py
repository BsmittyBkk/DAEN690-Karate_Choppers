#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import Libraries for use
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np
import os 
import glob
from sklearn.feature_selection import VarianceThreshold


# In[2]:


# Select 1 file to load in the column names
df1 = pd.read_csv('SimData_2023.06.13_09.48.56.csv')


# In[3]:


#get column names
column_names = list(df1.columns)
print(column_names)


# In[4]:


#delete df1 from above to save space
del df1


# In[5]:


#Jane's code for loading CSV's into one DF
#Path variable is unique to each user
path = r'/Volumes/Backup Plus/Brett DAEN 690/6:15:23-Flight-Data'
folder = glob.glob(os.path.join(path, "*.csv"))
df = (pd.read_csv(f, names = column_names, skiprows = 3, skipfooter=1) for f in folder)
df_comb = pd.concat(df,ignore_index=True)


# In[6]:


# Check the size of the current combined DF
print(df_comb.shape)


# In[7]:


#Check the data types
print(df_comb.dtypes.value_counts())


# In[8]:


# Check for null values
print(df_comb.isna().sum())
## The 30 missing values


# In[ ]:





# In[ ]:





# In[9]:


# Display the 30 missing values 
print(df_comb[df_comb['Elapsed Time'].isna()])
# From this, we can see these entries are null because the entry was a place holder to state the status of the sim
# Will drop these entries, no data is lost in this process


# In[10]:


#Drop the NaN entries, where the sim status was
df_comb.dropna(subset=['Elapsed Time'], inplace=True)


# In[11]:


# Check for null values again
print(df_comb.isna().sum())


# In[12]:


#Display columns where there are null values
print(df_comb.columns[df_comb.isna().sum() > 0])


# In[13]:


# Make plot of variables with null values
print(df_comb.isna().sum()[df_comb.isna().sum()>0].plot(kind='bar'))


# In[14]:


# Create the new table with date, start time, end time, and maneuver names (based off of Flight Log -- CS handtyped so check for errors!)
label_table = pd.DataFrame({
    'Date': ['2023-06-15', '2023-06-15', '2023-06-15', '2023-06-15', '2023-06-15', '2023-06-15', '2023-06-15', '2023-06-15', '2023-06-15', '2023-06-15', '2023-06-15', '2023-06-15', '2023-06-15', '2023-06-15', '2023-06-15', '2023-06-15', '2023-06-15', '2023-06-15'],  # Replace with actual dates of maneuvers
    'StartTime': ['13:22:15.0', '13:25:25.0', '13:29:25.0', '11:56:25.0', '11:58:03.0', '11:59:51.0', '16:10:04.0', '16:11:41.0', '16:14:20.0', '13:43:12.0', '13:44:10.0', '13:45:19.0', '12:08:11.0', '12:09:31.0', '12:10:51.0', '16:34:28.0', '16:35:06.0', '16:38:26.0'],  # Replace with actual start time of maneuvers
    'EndTime': ['13:22:25.0', '13:25:38.0', '13:29:40.0', '11:56:38.0', '11:58:24.0', '12:00:00.0', '16:10:12.0', '16:11:46.0', '16:14:29.0', '13:43:35.0', '13:44:18.0', '13:45:30.0', '12:08:35.0', '12:09:52.0', '12:11:18.0', '16:34:42.0', '16:35:27.0', '16:38:36.0'],  # Replace with actual end time of maneuvers
    'Label': ['Dynamic Rollover', 'Dynamic Rollover', 'Dynamic Rollover', 'Dynamic Rollover', 'Dynamic Rollover', 'Dynamic Rollover', 'Dynamic Rollover', 'Dynamic Rollover', 'Dynamic Rollover', 'LOW-G', 'LOW-G', 'LOW-G', 'LOW-G', 'LOW-G', 'LOW-G', 'LOW-G', 'LOW-G', 'LOW-G']  # Replace with maneuver names
})


# In[15]:


# Convert date, start time, and end time columns to datetime type
label_table['Date'] = pd.to_datetime(label_table['Date'])
label_table['StartTime'] = pd.to_datetime(label_table['StartTime'], format='%H:%M:%S.%f').dt.strftime('%H:%M:%S.%f')
label_table['EndTime'] = pd.to_datetime(label_table['EndTime'], format='%H:%M:%S.%f').dt.strftime('%H:%M:%S.%f')


# In[16]:


# Convert the time column on original df to correct format
df_comb['System UTC Time'] = pd.to_datetime(df_comb['System UTC Time'], format='%H:%M:%S.%f').dt.strftime('%H:%M:%S.%f')
# Convert the date column on original df to correct format
df_comb['Date'] = pd.to_datetime(df_comb['Date'])


# In[17]:


# Check the datetime conversions
print(df_comb.dtypes)
print(df_comb.dtypes.value_counts())


# In[18]:


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


# In[19]:


# Set NaN values in Label Row to 'Other'
df_comb['Label'] = df_comb['Label'].fillna('Other')


# In[21]:


# Look at entries where Transmission Oil Pressure Warning is NaN
print(df_comb[(df_comb['Transmission Oil Pressure Warning'].isnull()) & df_comb['Label'] == 'Dynamic Rollover'].head(20))
# None of the entries where Transmission Oil Pressure Warning are when a manuever is happening


# In[22]:


# Check for infinite values
number_inf = df_comb[df_comb == np.inf].count()


# In[25]:


# Looking at instances where inifinity values occur
print(number_inf[number_inf > 0])


# In[26]:


# Drop columns with the infinite value and NaN's
# First change inf to NaN's 
df_comb.replace([np.inf, -np.inf], np.nan, inplace=True)


# In[33]:


# Next gather all columns with NaN's and drop them
col_nan = [col for col in df_comb.columns if df_comb[col].isna().sum() > 0]

print(col_nan)


# In[35]:


# Drop the NaN columns
df_comb.drop(col_nan, axis=1, inplace=True)


# In[36]:


#Check the shape to make sure the columns were dropped
print(df_comb.shape)


# In[45]:


# Check for near zero and zero variance columns
# Create a DF that does not have anything but numeric data to check the variance
columns_with_one_unique_value = []
for column in df_comb.columns:
    if df_comb[column].nunique() == 1:
        columns_with_one_unique_value.append(column)
print(columns_with_one_unique_value)


# In[46]:


# Create DF without low variance columns
df_reduced = df_comb.drop(columns_with_one_unique_value, axis=1)


# In[47]:


# Check the shape of the new DF
print(df_reduced.shape)


# In[50]:


# Create a correlation heat map
fig, ax = plt.subplots(figsize=(15,15))

data_corr_plot2 = sb.heatmap(df_reduced.corr(), cmap = 'RdBu', ax = ax)

print(data_corr_plot2)


# In[51]:


## Checking top correlated variables
df_reduced_corr = df_reduced.corr().abs() 
high_corr = (df_reduced_corr.where(np.triu(np.ones(df_reduced_corr.shape), k=1).astype(bool)).stack().sort_values(ascending=False))    


# In[56]:


# Display top 30 highly correlated variables
print(high_corr.head(30))


# In[ ]:




