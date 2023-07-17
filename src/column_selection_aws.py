import pandas as pd
import numpy as np
import pickle

all_cols = ['Elapsed Time', 'Date', 'System UTC Time', 'Latitude', 'Longitude', 'Altitude(MSL)', 'Altitude(AGL)', 'Heading(mag)', 'Heading(true)', 'Pitch', 'Roll', 'Yaw', 'Alpha', 'Beta', 'Groundspeed', 'Airspeed(Ind)', 'Airspeed(True)', 'Vert. Speed', 'Baro Altitude Pilot', 'Baro Setting Pilot', 'Baro Altitude Copilot', 'Baro Setting Copilot', 'Radio Altitude Pilot', 'Radio Altitude Copilot', 'Wheels On Ground-[0]', 'Wheels On Ground-[1]', 'Wheels On Ground-[2]', 'Compass Heading', 'Ground Track Pilot', 'Ground Track Copilot', 'Roll Rate', 'Pitch Rate', 'Yaw Rate', 'Turn Rate', 'Sideslip Angle', 'Angle of Attack', 'FMS Vert Flight Path Angle', 'Pitch Path', 'Heading Path', 'Flight Path Angle - VV-[0]', 'Flight Path Angle - VV-[1]', 'Flight Path Angle - VV-[2]', 'Ground Track - VV-[0]', 'Ground Track - VV-[1]', 'Ground Track - VV-[2]', 'Roll Acceleration', 'Pitch Acceleration', 'Yaw Acceleration', 'Acceleration in Latitude', 'Acceleration in Normal', 'Acceleration in Longitudinal', 'Cyclic Pitch Pos-[0]', 'Cyclic Roll Pos-[0]', 'Collective Pos-[0]', 'Pedal Pos', 'Left Brake Pos', 'Right Brake Pos', 'Throttle Position', 'Wheel Brake', 'Gear Position (Down)', 'Rotor Brake', 'Fuel Weight', 'Gross Weight', 'Fuel Flow-[0]', 'Fuel Flow-[1]', 'Fuel Pressure-[0]', 'Fuel Pressure-[1]', 'Oil Pressure-[0]', 'Oil Pressure-[1]', 'Oil Temp', 'Engine Temp-[0]', 'Engine Temp-[1]', 'Engine Torque (N1)', 'Engine Torque (N2)', 'Engine speeds (N1)-[0]', 'Engine speeds (N1)-[1]', 'Engine speeds (N2)-[0]', 'Engine speeds (N2)-[1]', 'Engine speeds (NG2)', 'Engine speeds (NG1)', 'Engine speeds (NP2)-[0]', 'Engine speeds (NP2)-[1]', 'Engine speeds (NP1)-[0]', 'Engine speeds (NP1)-[1]', 'Rotor RPM-[0]', 'Rotor RPM Value-[0]', 'Rotor RPM Value-[1]', 'Rotor Torque-[0]', 'Engine Power-[0]', 'Engine Power-[1]', 'Electrical-[0]', 'Electrical-[1]', 'T5 Temp ITT-[0]', 'T5 Temp ITT-[1]', 'Lateral Cyclic Eng Rotor-Disc-[0]', 'Longitudinal Cyclic Eng Rotor-Disc-[0]', 'Lateral Disc Tilt Cyclic Deflection-[0]', 'Lateral Disc Tilt Cyclic Deflection-[1]', 'Lateral Disc Tilt Cyclic Deflection-[2]', 'Lateral Disc Tilt Cyclic Deflection-[3]', 'Longitudinal Disc Tilt Cyclic Deflection-[0]', 'Longitudinal Disc Tilt Cyclic Deflection-[1]', 'Longitudinal Disc Tilt Cyclic Deflection-[2]', 'Longitudinal Disc Tilt Cyclic Deflection-[3]', 'Induced Velo Behind Disc-[0]', 'Induced Velo Behind Disc-[1]', 'Total Engine Thrust-[0]', 'Total Engine Thrust-[1]', 'Main Rotor Pos', 'Main Rotor Angle Slow', 'Swashplate Blade Gain', 'Swashplate Rotor 000', 'Swashplate Rotor 072', 'Swashplate Rotor 144', 'Swashplate Rotor 216', 'Swashplate Rotor 288', 'Swashplate Phase', 'FD pitch', 'FD roll', 'ATT Status', 'SAS Status', 'Autopilot Modes', 'Autopilot Engaged', 'ALTP Status', 'ALT Status', 'HDG Status', 'IAS Status', 'VS Status', 'APPR Status', 'NAV Status', 'VNAV Status', 'RALT Status', 'VHLD Status', 'DECEL Status', 'BC Status', 'TOGA Status', 'STBY Status', 'CPL Status', 'AP1 Status', 'AP2 Status', 'Flight ID', 'GPS 1 NAV ID', 'GPS 1 DME Time', 'GPS 1 DME Distance', 'GPS 1 DME Speed', 'NAV 1 NAV ID', 'NAV 1 DME Time', 'NAV 1 DME Distance', 'NAV 1 DME Speed', 'NAV 2 NAV ID', 'NAV 2 DME Time', 'NAV 2 DME Distance', 'NAV 2 DME Speed', 'FMS Waypoints', 'HSI Source Selection', 'Nav1 Hor Deviation', 'Nav1 Ver Deviation', 'Nav2 Hor Deviation', 'Nav2 Ver Deviation', 'GPS Hor Deviation', 'GPS Ver Deviation', 'VHF Com1 Freq', 'VHF Com2 Freq', 'VHF Nav1 Freq', 'VHF Nav2 Freq', 'Cyclic Float', 'Cyclic Cargo', 'Outside Air Temp', 'Wind Speed(True)', 'Wind Direction(Mag)', 'Elevation', 'Actual Visibility', 'Cloud Type 0', 'Cloud Type 1', 'Cloud Type 2', 'Cloud Coverage 0', 'Cloud Coverage 1', 'Cloud Coverage 2', 'Cloud Base MSL 0', 'Cloud Base MSL 1', 'Cloud Base MSL 2', 'Cloud Tops MSL 0', 'Cloud Tops MSL 1', 'Cloud Tops MSL 2', 'Altimeter Indicator Failure', 'Airspeed Indicator Failure', 'Battery Low Volt Warning', 'EGPWS Alert', 'Engine Chip Detected', 'Fuel Pump 0 Failure Warning', 'Fuel Pump 1 Failure Warning', 'Fuel Low Warning', 'Generator 0 Failure Warning', 'Generator 1 Failure Warning', 'Hydraulic System Warning', 'Rotor Low RPM Warning', 'Rotor High RPM Warning', 'Tail Rotor Chip Warning', 'Transmission Chip Warning', 'Transmission Oil Temp Warning', 'Transmission Oil Pressure Warning']

col_dtypes = ['float64', 'str', 'str', 'str', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'str', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'str', 'str', 'float64', 'float64', 'float64', 'str', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'str', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'str', 'str', 'str', 'str', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64']

dtypes = dict(zip(all_cols, col_dtypes))

not_in_aws= pd.read_excel('../../CSV/missing_field_equivalencies.xlsx')

# Extract the values from the 'Field Names' column and convert to a list
not_aws = not_in_aws['Field Name'].tolist()

aws_cols = [x for x in all_cols if x not in not_aws]

aws_dtype = {key: value for key, value in dtypes.items() if key in aws_cols}

df = pd.read_csv("../../CSV/SimData_2023.06.13_09.48.56.csv",usecols=aws_cols, names=aws_cols,
                  skiprows=[0, 1, 2], skip_blank_lines=True)
df1 = df.copy()

print(f"All columns in original dataset: \n{df.columns.tolist()})")
print(f"Initial shape of the DataFrame: {df.shape}")

# find non_numeric columns and remove the columns not needed
non_numeric_columns = df.select_dtypes(exclude='number').columns.tolist()
print(f"Non-numeric columns: {non_numeric_columns}")
# Non-numeric columns: ['Date', 'System UTC Time', 'Latitude', 'Autopilot Modes', 
# 'Flight ID', 'GPS 1 NAV ID', 'NAV 1 NAV ID', 'FMS Waypoints', 'Actual Visibility', 
# 'Cloud Type 0', 'Cloud Type 1', 'Cloud Type 2']

# keep date and time for labelling later
keep = ['Date', 'System UTC Time']
no_num = [col for col in non_numeric_columns if col not in keep]
# add latitude and longitude and elapsed time to the list of variable to be removed
no_num.extend(['Longitude'])
print(f"Columns being dropped: {no_num}")
# Columns being dropped: ['Latitude', 'Autopilot Modes', 'Flight ID', 'GPS 1 NAV ID', 
# 'NAV 1 NAV ID', 'FMS Waypoints', 'Actual Visibility', 'Cloud Type 0', 'Cloud Type 1', 
# 'Cloud Type 2', 'Longitude', 'Elapsed Time']
# remove unwanted columns
df.drop(columns=no_num, inplace=True)

# Calc Variance for each column that is not date or time
variance = df.iloc[:, 3:].var()

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
df.drop(columns=near_zero_variance_columns, inplace=True)

correlation_matrix = df.iloc[:, 3:].corr()

# Identify correlated columns
corr_threshold = .9
correlated_columns = []
for col1 in correlation_matrix.columns:
    for col2 in correlation_matrix.columns:
        if col1 != col2 and np.abs(correlation_matrix.loc[col1, col2]) > corr_threshold:
            correlated_columns.append(col1)

# Drop correlated columns from the dataframe
df.drop(correlated_columns, axis=1, inplace=True)

print(f"Final shape of the DataFrame: {df.shape}")
# Final shape of the DataFrame: (27572, 118)

use_cols = df.columns.tolist()

print(f"Final column list with low variance columns removed: \n{use_cols}")
# Final column list with low variance and high correlation columns removed:
# ['Elapsed Time', 'Date', 'System UTC Time', 'Heading(mag)', 'Vert. Speed', 'Compass Heading', 'Roll Rate', 'Pitch Rate', 'Pitch Path', 'Heading Path', 'Flight Path Angle - VV-[0]', 'Ground Track - VV-[0]', 'Roll Acceleration', 'Pitch Acceleration', 'Yaw Acceleration', 'Pedal Pos', 'Induced Velo Behind Disc-[0]', 'Induced Velo Behind Disc-[1]', 'Main Rotor Pos', 'Main Rotor Angle Slow', 'Swashplate Rotor 000', 'Swashplate Rotor 072', 'Swashplate Rotor 144', 'Swashplate Rotor 216', 'Swashplate Rotor 288', 'Swashplate Phase', 'GPS 1 DME Distance', 'GPS 1 DME Speed', 'NAV 1 DME Time', 'NAV 1 DME Speed', 'NAV 2 NAV ID', 'GPS Hor Deviation']

with open('./use_cols_aws.pkl', 'wb') as f:
    pickle.dump(use_cols, f)
