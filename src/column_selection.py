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

print(f"Final shape of the DataFrame: {df1.shape}")
# Final shape of the DataFrame: (27572, 118)

use_cols = ['Elapsed Time']
use_cols.extend(df1.columns.tolist())
print(f"Final column list with low variance columns removed: \n{use_cols}")
# Final column list with low variance columns removed:
# ['Elapsed_Time', 'Date', 'System UTC Time', 'Altitude(MSL)', 'Altitude(AGL)', 'Heading(mag)', 'Heading(true)', 'Pitch', 'Roll', 
# 'Yaw', 'Alpha', 'Beta', 'Groundspeed', 'Airspeed(Ind)', 'Airspeed(True)', 'Vert. Speed', 'Baro Altitude Pilot', 
# 'Baro Altitude Copilot', 'Radio Altitude Pilot', 'Radio Altitude Copilot', 'Wheels On Ground-[0]', 'Wheels On Ground-[1]', 
# 'Wheels On Ground-[2]', 'Compass Heading', 'Ground Track Pilot', 'Ground Track Copilot', 'Roll Rate', 'Pitch Rate', 
# 'Yaw Rate', 'Turn Rate', 'Sideslip Angle', 'Angle of Attack', 'Pitch Path', 'Heading Path', 'Flight Path Angle - VV-[0]', 
# 'Ground Track - VV-[0]', 'Roll Acceleration', 'Pitch Acceleration', 'Yaw Acceleration', 'Cyclic Pitch Pos-[0]', 
# 'Cyclic Roll Pos-[0]', 'Collective Pos-[0]', 'Pedal Pos', 'Throttle Position', 'Fuel Weight', 'Gross Weight', 
# 'Oil Pressure-[0]', 'Oil Pressure-[1]', 'Engine Temp-[0]', 'Engine Temp-[1]', 'Engine Torque (N1)', 'Engine Torque (N2)', 
# 'Engine speeds (N1)-[0]', 'Engine speeds (N1)-[1]', 'Engine speeds (N2)-[0]', 'Engine speeds (N2)-[1]', 
# 'Engine speeds (NG2)', 'Engine speeds (NG1)', 'Engine speeds (NP2)-[0]', 'Engine speeds (NP2)-[1]', 'Engine speeds (NP1)-[0]', 
# 'Engine speeds (NP1)-[1]', 'Rotor RPM-[0]', 'Rotor RPM Value-[0]','Rotor RPM Value-[1]', 'Rotor Torque-[0]', 
# 'Engine Power-[0]', 'Engine Power-[1]', 'T5 Temp ITT-[0]', 'T5 Temp ITT-[1]', 'Lateral Cyclic Eng Rotor-Disc-[0]', 
# 'Longitudinal Cyclic Eng Rotor-Disc-[0]', 'Lateral Disc Tilt Cyclic Deflection-[0]', 'Longitudinal Disc Tilt Cyclic Deflection-[0]', 
# 'Induced Velo Behind Disc-[0]', 'Induced Velo Behind Disc-[1]', 'Total Engine Thrust-[0]', 'Total Engine Thrust-[1]', 'Main Rotor Pos', 
# 'Main Rotor Angle Slow', 'Swashplate Rotor 000', 'Swashplate Rotor 072', 'Swashplate Rotor 144', 'Swashplate Rotor 216', 
# 'Swashplate Rotor 288', 'Swashplate Phase', 'FD pitch', 'FD roll', 'NAV Status', 'STBY Status', 'CPL Status',
# 'GPS 1 DME Distance', 'GPS 1 DME Speed', 'NAV 1 DME Time', 'NAV 1 DME Distance', 'NAV 1 DME Speed', 'NAV 2 NAV ID', 'HSI Source Selection', 
# 'Nav1 Hor Deviation', 'GPS Hor Deviation', 'GPS Ver Deviation', 'VHF Com1 Freq', 'VHF Com2 Freq', 'VHF Nav1 Freq', 'VHF Nav2 Freq', 
# 'Cyclic Cargo', 'Outside Air Temp', 'Wind Speed(True)', 'Wind Direction(Mag)', 'Elevation', 'Cloud Coverage 0', 
# 'Cloud Coverage 2', 'Cloud Base MSL 0', 'Cloud Base MSL 1', 'Cloud Base MSL 2', 'Cloud Tops MSL 0', 'Cloud Tops MSL 1', 'Cloud Tops MSL 2']


with open('./use_cols.pkl', 'wb') as f:
    pickle.dump(use_cols, f)
