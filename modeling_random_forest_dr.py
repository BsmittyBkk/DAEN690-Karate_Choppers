import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, make_scorer, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

# this is the path to your pickle file (should be the same location as CSVs)
path = r'../CSV'

# the below function verifies that the dataframe you are working with is the same shape as the anticipated dataframe
def test_dataframe_shape():
    # load the dataframe to be tested
    with open(f'{path}/working_df2.pkl', 'rb') as file:
        df = pickle.load(file)
    # Perform the shape validation
    # assert df.shape == (258905, 118)
    return df

# working dataframe that has 'Label', 'Dynamic Rollover', 'LOW-G' as the final 3 columns
df = test_dataframe_shape().reset_index(drop=True)

## to test on Dynamic Rollover
df = df.drop(columns=['Label', 'LOW-G', 'Swashplate Rotor 216', 'Swashplate Phase', 'Main Rotor Angle Slow', 'Swashplate Rotor 072'])
# Swashplate Rotor 216: 0.0
# Swashplate Phase: 0.0
# Main Rotor Angle Slow: 0.0
# Swashplate Rotor 072: 0.0

## to test on LOW-G
# df = df.drop(columns=['Label', 'Dynamic Rollover'])

# define X and y Dynamic Rollover
X = df.drop('Dynamic Rollover', axis=1)
y = df['Dynamic Rollover']

# define X and y for LOW-G
# X = df.drop('LOW-G', axis=1)
# y = df['LOW-G']

# create training set and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
# y_test: 
# Dynamic Rollover
# 0    51270
# 1      511
# Name: count, dtype: int64

params = {
    'rf__n_estimators': [50],  # 100, 200
    'rf__max_depth': [None],  # 5, 10
    'rf__min_samples_split': [2],  # 5, 10
    'rf__min_samples_leaf': [1],  # 2, 4
    'rf__max_features': ['log2'],  # 'sqrt'
    'rf__bootstrap': [False],  # True
    'rf__class_weight': ['balanced'],
    'rf__random_state': [42],
    'rf__n_jobs': [-1]
}

# create a pipeline
pipeline = Pipeline([
    ('rf', RandomForestClassifier())
])

f1_scorer = make_scorer(f1_score)
strat_k_fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid_search = GridSearchCV(estimator=pipeline, param_grid=params, cv=strat_k_fold, scoring=f1_scorer)
grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
best_model = grid_search.best_estimator_  

y_pred = best_model.predict(X_test)

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(best_params)

# Access and sort feature importances
importances = best_model.named_steps['rf'].feature_importances_
sorted_indices = importances.argsort()[::-1]

# Retrieve feature names
feature_names = list(X_train.columns)

# Print the most important variables
print("Most important variables:")
for i in sorted_indices:
    print(f"{feature_names[i]}: {importances[i]}")
# Most important variables:
# Heading(mag): 0.1333131638933488
# Vert. Speed: 0.11467235716814925
# Pitch Rate: 0.08352635652141906
# Ground Track - VV-[0]: 0.07916883017326566
# Pitch Path: 0.06812558393239819
# NAV 1 DME Time: 0.052290925703181224
# GPS 1 DME Speed: 0.047630667898031824
# Main Rotor Pos: 0.04753005588430459
# Roll Rate: 0.0391460816982194
# Induced Velo Behind Disc-[1]: 0.035936186446882484
# NAV 1 DME Speed: 0.03189322373131597
# Roll Acceleration: 0.031576941072159005
# Compass Heading: 0.030373178934428733
# Swashplate Rotor 000: 0.027274968668887904
# Swashplate Rotor 144: 0.024423913534007625
# Heading Path: 0.023596783091042645
# Pedal Pos: 0.019974152832028687
# GPS 1 DME Distance: 0.019294332239174913
# Yaw Acceleration: 0.019257277527676055
# Induced Velo Behind Disc-[0]: 0.01903254536940032
# Swashplate Rotor 288: 0.01565642525317144
# Flight Path Angle - VV-[0]: 0.013726643211687735
# GPS Hor Deviation: 0.011604542503929647
# Pitch Acceleration: 0.007547200927142835
# NAV 2 NAV ID: 0.003427661784746056
