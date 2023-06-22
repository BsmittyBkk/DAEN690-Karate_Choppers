#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd


# In[7]:


import numpy as np


# In[8]:


import matplotlib 


# In[9]:


import matplotlib.pyplot as plt


# In[10]:


import seaborn as sns


# In[11]:


import warnings


# In[17]:


warnings.filterwarnings('ignore')


# In[18]:


RCdf_20230613_094856 = pd.read_csv('SimData_20230613_094856.csv', low_memory=False)


# In[19]:


#RCdf_20230613_094856 


# In[20]:


RCdf_20230613_094856.shape


# In[21]:


RCdf_20230613_094856.describe(include=object)


# In[22]:


RCdf_20230613_094856.dtypes.value_counts()


# In[23]:


RCdf_20230613_094856.info()


# In[ ]:





# In[24]:


RCdata_0216_91311 = pd.read_csv('SimData_20230216_091311roar.csv')


# In[25]:


#pd.set_option('display.max_rows', None)


# In[26]:


#pd.set_option("display.precision", 2)


# In[27]:


RCdata_0216_91311


# In[28]:


RCdata_0216_91311.head()


# In[29]:


RC_20230601_130640 = pd.read_csv('SimData_20230601_130640roar.csv', low_memory=False)


# In[30]:


RC_20230601_130640 


# In[31]:


RCdtypes_20230601_130640 = RC_20230601_130640.dtypes


# In[32]:


RCdtypes_20230601_130640 


# In[33]:


from pathlib import Path


# In[34]:


print(Path.cwd())


# # Count of data type and visualization

# In[35]:


RC_20230601_130640.dtypes.value_counts()


# In[36]:


type(RC_20230601_130640)


# In[37]:


# Exploring how much data the df contains
# Len- determine the number of rows
# datasetname.shape - The shape attribute of the dataframe
# shows its dimensionality by showing rows and columns.


# In[38]:


RC_20230601_130640.shape


# In[39]:


RC_20230601_130640.info()


# In[40]:


RC_20230601_130640.describe(include=object)


# In[41]:


# print(RC_20230601_130640.describe())


# In[42]:


rows_without_missing_data = RC_20230601_130640.dropna()


# In[43]:


rows_without_missing_data.shape


# In[44]:


#Include this line to show plots directly in the notebook:
# %matplotlib inline


# In[45]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[46]:


RC_20230601_130640["Pitch"].value_counts().head(10).plot(kind="bar")


# In[47]:


# box and whisker plot


# In[48]:


import matplotlib.pyplot as plt


# In[49]:


RC_20230601_130640.plot(kind='box', subplots=False, layout=(191,4), sharex=False, sharey=False)
plt.show()


# In[50]:


# create a histogram 


# In[51]:


RC_20230601_130640.hist()


# In[52]:


plt.show()


# In[53]:


scatter_matrix(RC_20230601_130640)
pyplot.show()


# # Correlation and heatmap

# In[55]:


corr = RC_20230601_130640.corr()


# In[58]:


corr


# In[56]:


mask = np.triu(np.ones_like(corr, dtype=bool))


# In[57]:


sns.heatmap(corr, mask=mask, center=0,
            square=True,)


# # Concatenate csv files

# In[59]:


import glob


# In[60]:


import os


# In[63]:


KCdf = pd.read_csv('C:\Rprojects\ROARCSVs\KarateChoppersManeuvers-Datasets\SimData_20230615_160207.csv')


# In[68]:


#units = KCdf.iloc[0]


# In[69]:


column_names = list(KCdf.columns)


# In[71]:


#column_names


# In[77]:


path = r'C:\Rprojects\ROARCSVs\KarateChoppersManeuvers-Datasets'


# In[78]:


folder = glob.glob(os.path.join(path, "*.csv"))


# In[79]:


RCdata = (pd.read_csv(f, names = column_names, skiprows=1) for f in folder)


# In[80]:


RCdata_concat = pd.concat(RCdata, ignore_index=True)


# In[84]:


RCdata_concat.info()


# In[85]:


RCdata_concat.shape


# In[87]:


rows_without_missing_data = RCdata_concat.dropna()


# In[89]:


rows_without_missing_data


# In[90]:


RCdata_concat.shape


# In[91]:


RCdata_concat.describe(include=object)


# In[98]:


RCdata_concat.head()


# In[103]:


RCdata_concat.describe()


# In[106]:


print ("The total number of elements in our object is:")
RCdata_concat.size


# In[3]:


print(RCdata_concat.isnull().sum())


# In[4]:


print(RCdata_concat.nunique())


# In[ ]:





# In[109]:


print ("The actual data in our data frame is:")
print(RCdata_concat.values)


# In[111]:


RCdata_concat.std()
#print(RCdata_concat.std)


# In[ ]:




