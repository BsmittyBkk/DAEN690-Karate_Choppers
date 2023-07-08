import pickle
import pytest
import pandas as pd

# this is the path to your pickle file (should be the same location as CSVs)
path = r'../CSV'

# the below function verifies that the dataframe you are working with is the same shape as the anticipated dataframe
def test_dataframe_shape():
    # load the dataframe to be tested
    with open(f'{path}/working_df.pkl', 'rb') as file:
        df = pickle.load(file)
    # Perform the shape validation
    assert df.shape == (575920, 118)
    return df

# working dataframe that has 'Label', 'Dynamic Rollover', 'LOW-G' as the final 3 columns
df = test_dataframe_shape().reset_index(drop=True)

## to test on Dynamic Rollover
# df = df.drop(columns=['Label', 'LOW-G'])
## to test on LOW-G
# df = df.drop(columns=['Label', 'Dynamic Rollover'])
