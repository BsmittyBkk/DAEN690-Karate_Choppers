import pandas as pd
import numpy as np
import glob
import os
import pickle


df1 = pd.read_csv("../../CSV/SimData_2023.06.13_09.48.56.csv",
                  skiprows=[1, 2], skipfooter=1, engine='python')

print(f"All columns in original dataset: \n{df1.columns.tolist()})")
print(f"Initial shape of the DataFrame: {df1.shape}")

# find non_numeric columns and remove the columns not needed
non_numeric_columns = df1.select_dtypes(exclude='number').columns.tolist()
print(f"Non-numeric columns: {non_numeric_columns}")
# Non-numeric columns: ['Date', 'System UTC Time', 'Latitude', 'Autopilot Modes', 
# 'Flight ID', 'GPS 1 NAV ID', 'NAV 1 NAV ID', 'FMS Waypoints', 'Actual Visibility', 
# 'Cloud Type 0', 'Cloud Type 1', 'Cloud Type 2']

# keep date and time for labelling later
keep = ['Date', 'System UTC Time']
no_num = [col for col in non_numeric_columns if col not in keep]
# add latitude and longitude and elapsed time to the list of variable to be removed
no_num.extend(['Longitude', 'Elapsed Time', 'GPS 1 DME Time'])
print(f"Columns being dropped: {no_num}")
# Columns being dropped: ['Latitude', 'Autopilot Modes', 'Flight ID', 'GPS 1 NAV ID', 
# 'NAV 1 NAV ID', 'FMS Waypoints', 'Actual Visibility', 'Cloud Type 0', 'Cloud Type 1', 
# 'Cloud Type 2', 'Longitude', 'Elapsed Time']
# remove unwanted columns
df1.drop(columns=no_num, inplace=True)

# Calc Variance for each column that is not date or time
variance = df1.iloc[:, 2:].var()

# Setting a threshold for variance -- thoughts?
threshold = 0.01

# Cols with near zero variance
near_zero_variance_columns = variance[variance <= threshold].index.tolist()
print(f"Columns with almost zero variance, includes those with only one value: \n{near_zero_variance_columns}")
# Columns with almost zero variance, includes those with only one value:
# ['Baro Setting Pilot', 'Baro Setting Copilot', 'FMS Vert Flight Path Angle', 'Flight Path Angle - VV-[1]', 
# 'Flight Path Angle - VV-[2]', 'Ground Track - VV-[1]', 'Ground Track - VV-[2]', 'Acceleration in Latitude', 
# 'Acceleration in Normal', 'Acceleration in Longitudinal', 'Left Brake Pos', 'Right Brake Pos', 'Wheel Brake', 
# 'Gear Position (Down)', 'Rotor Brake', 'Fuel Flow-[0]', 'Fuel Flow-[1]', 'Fuel Pressure-[0]', 'Fuel Pressure-[1]', 
# 'Oil Temp', 'Electrical-[0]', 'Electrical-[1]', 'Lateral Disc Tilt Cyclic Deflection-[1]', 
# 'Lateral Disc Tilt Cyclic Deflection-[2]', 'Lateral Disc Tilt Cyclic Deflection-[3]', 
# 'Longitudinal Disc Tilt Cyclic Deflection-[1]', 'Longitudinal Disc Tilt Cyclic Deflection-[2]', 
# 'Longitudinal Disc Tilt Cyclic Deflection-[3]', 'Swashplate Blade Gain', 'ATT Status', 'SAS Status', 
# 'Autopilot Engaged','ALTP Status', 'ALT Status', 'HDG Status', 'IAS Status', 'VS Status', 'APPR Status', 
# 'VNAV Status', 'RALT Status', 'VHLD Status', 'DECEL Status', 'BC Status', 'TOGA Status', 'AP1 Status', 'AP2 Status', 
# 'NAV 2 DME Time', 'NAV 2 DME Distance', 'NAV 2 DME Speed', 'Nav1 Ver Deviation', 'Nav2 Hor Deviation', 
# 'Nav2 Ver Deviation', 'Cyclic Float', 'Cloud Coverage 1', 'Altimeter Indicator Failure', 'Airspeed Indicator Failure', 
# 'Battery Low Volt Warning', 'EGPWS Alert', 'Engine Chip Detected', 'Fuel Pump 0 Failure Warning', 
# 'Fuel Pump 1 Failure Warning', 'Fuel Low Warning', 'Generator 0 Failure Warning', 'Generator 1 Failure Warning', 
# 'Hydraulic System Warning', 'Rotor Low RPM Warning', 'Rotor High RPM Warning', 'Tail Rotor Chip Warning', 
# 'Transmission Chip Warning', 'Transmission Oil Temp Warning', 'Transmission Oil Pressure Warning']
df1.drop(columns=near_zero_variance_columns, inplace=True)

correlation_matrix = df1.iloc[:, 2:].corr()

# Identify correlated columns
corr_threshold = .9
correlated_columns = []
for col1 in correlation_matrix.columns:
    for col2 in correlation_matrix.columns:
        if col1 != col2 and np.abs(correlation_matrix.loc[col1, col2]) > corr_threshold:
            correlated_columns.append(col1)

correlated_columns.append('VHF Com1 Freq')
# Drop correlated columns from the dataframe
df1.drop(correlated_columns, axis=1, inplace=True)

print(f"Final shape of the DataFrame: {df1.shape}")
# Final shape of the DataFrame: (27572, 118)

use_cols = ['Elapsed Time']
use_cols.extend(df1.columns.tolist())
print(f"Final column list with low variance columns removed: \n{use_cols}")
# Final column list with low variance and high correlation columns removed:
# ['Elapsed Time', 'Date', 'System UTC Time', 'Heading(mag)', 'Vert. Speed', 'Compass Heading', 'Roll Rate', 'Pitch Rate', 'Pitch Path', 'Heading Path', 'Flight Path Angle - VV-[0]', 'Ground Track - VV-[0]', 'Roll Acceleration', 'Pitch Acceleration', 'Yaw Acceleration', 'Pedal Pos', 'Induced Velo Behind Disc-[0]', 'Induced Velo Behind Disc-[1]', 'Main Rotor Pos', 'Main Rotor Angle Slow', 'Swashplate Rotor 000', 'Swashplate Rotor 072', 'Swashplate Rotor 144', 'Swashplate Rotor 216', 'Swashplate Rotor 288', 'Swashplate Phase', 'GPS 1 DME Distance', 'GPS 1 DME Speed', 'NAV 1 DME Time', 'NAV 1 DME Speed', 'NAV 2 NAV ID', 'GPS Hor Deviation']

with open('./use_cols.pkl', 'wb') as f:
    pickle.dump(use_cols, f)
