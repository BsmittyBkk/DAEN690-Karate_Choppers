import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# this is the path to your pickle file (should be the same location as CSVs)
path = r'../CSV'

# the below function verifies that the dataframe you are working with is the same shape as the anticipated dataframe
# def test_dataframe_shape():
#     # load the dataframe to be tested

#     # Perform the shape validation
#     # assert df.shape == (27572, 31)
#     return df

# working dataframe that has 'Label', 'Dynamic Rollover', 'LOW-G' as the final 3 columns
# df = test_dataframe_shape().reset_index(drop=True)
with open(f'{path}/working_df_dr.pkl', 'rb') as file:
    df = pickle.load(file)

df['Heading(mag)'] = df['Heading(mag)'].astype('float64')

## to test on Dynamic Rollover
df = df.drop(columns=['LOW-G'])
## to test on LOW-G
# df = df.drop(columns=['Label', 'Dynamic Rollover'])


