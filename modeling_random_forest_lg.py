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
    with open(f'{path}/working_df_aws.pkl', 'rb') as file:
        df = pickle.load(file)
    # Perform the shape validation
    # assert df.shape == (258905, 118)
    return df

# working dataframe that has 'Label', 'Dynamic Rollover', 'LOW-G' as the final 3 columns
df = test_dataframe_shape().reset_index(drop=True)

## to test on Low-G
df = df.drop(columns=['Dynamic Rollover', 'NAV 2 DME Time', 'GPS 1 DME Time', 'NAV 2 NAV ID', 'GPS 1 NAV ID', 'FMS Waypoints'])
# Main Rotor Angle Slow: 0.0
# Swashplate Rotor 072: 0.0
# Swashplate Rotor 288: 0.0

# define X and y Dynamic Rollover
X = df.drop('LOW-G', axis=1)
y = df['LOW-G']

# create training set and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)


params = {
    'rf__n_estimators': [50],  # 100, 200
    'rf__max_depth': [None],  # 5, 10
    'rf__min_samples_split': [2],  # 5, 10
    'rf__min_samples_leaf': [1],  # 2, 4
    'rf__max_features': ['log2'],  # 'sqrt'
    'rf__bootstrap': [True],  # False
    'rf__class_weight': ['balanced'],
    'rf__random_state': [42],
    'rf__n_jobs': [-1]
}

# create a pipeline to test the model
pipeline = Pipeline([
    ('rf', RandomForestClassifier())
])

# create fl_scorer to use in the grid search
f1_scorer = make_scorer(f1_score)

# set up k-fold cross validation
strat_k_fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# instantiate the grid search loading pipeline, parameters, k-fold, and scorer
grid_search = GridSearchCV(estimator=pipeline, param_grid=params, cv=strat_k_fold, scoring=f1_scorer)
# fit the grid search
grid_search.fit(X_train, y_train)

# save the best parameters from the grid search and the model with those parameters
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_  

# predict on the test set with the best model
y_pred = best_model.predict(X_test)

# print outputs
print(classification_report(y_test, y_pred))
#               precision    recall  f1-score   support

#            0       1.00      1.00      1.00     51437
#            1       1.00      1.00      1.00       344

#     accuracy                           1.00     51781
#    macro avg       1.00      1.00      1.00     51781
# weighted avg       1.00      1.00      1.00     51781
print(confusion_matrix(y_test, y_pred))
# [[51437     0]
#  [    1   343]]
print(best_params)
# 'rf__bootstrap': True, 'rf__class_weight': 'balanced', 'rf__max_depth': None, 'rf__max_features': 'sqrt', 'rf__min_samples_leaf': 1, 'rf__min_samples_split': 2, 'rf__n_estimators': 50, 'rf__n_jobs': -1, 'rf__random_state': 42

# access and sort feature importances
importances = best_model.named_steps['rf'].feature_importances_
sorted_indices = importances.argsort()[::-1]

# retrieve feature names
feature_names = list(X_train.columns)

# print the most important variables
print("Most important variables:")
for i in sorted_indices:
    print(f"{feature_names[i]}: {importances[i]}")
# Most important variables:
# Heading(mag): 0.18877636848104615
# Transmission Chip Warning: 0.16453997340913962
# Ground Track - VV-[2]: 0.14508524708793535
# Flight Path Angle - VV-[0]: 0.10296199009725504
# Transmission Oil Temp Warning: 0.07965926418227508
# Turn Rate: 0.07495080551143295
# Baro Setting Pilot: 0.05610011618464458
# Right Brake Pos: 0.0524722462763778
# TOGA Status: 0.05057122210309971
# Acceleration in Normal: 0.026162563811435585
# Ground Track Copilot: 0.01756437605402702
# NAV 2 DME Speed: 0.016166003845747526
# NAV 2 DME Distance: 0.014527569545151654
# Yaw Rate: 0.005808970336183176
# Yaw Acceleration: 0.0018848388599306738
# Flight Path Angle - VV-[1]: 0.001119721103937614
# Acceleration in Latitude: 0.0007019428298626385
# AP1 Status: 0.0005950836101104052
# Flight Path Angle - VV-[2]: 0.00035169667010582405
# Tail Rotor Chip Warning: 2.0295617887153638e-13
# Nav1 Ver Deviation: 9.864319043187931e-14
