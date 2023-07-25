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
    with open(f'{path}/dynamic_rollover.pkl', 'rb') as file:
        df = pickle.load(file)
    # Perform the shape validation
    # assert df.shape == (258905, 118)
    return df

# working dataframe that has 'Label', 'Dynamic Rollover', 'LOW-G' as the final 3 columns
df = test_dataframe_shape().reset_index(drop=True)

## to test on Dynamic Rollover
## to test on LOW-G
# df = df.drop(columns=['Label', 'Dynamic Rollover'])

# define X and y Dynamic Rollover
X = df.drop('label_Dynamic Rollover', axis=1)
y = df['label_Dynamic Rollover']

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
    'rf__min_samples_split': [10],  # 5, 10
    'rf__min_samples_leaf': [2],  # 2, 4
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
#               precision    recall  f1-score   support

#            0       1.00      1.00      1.00     51270
#            1       1.00      1.00      1.00       511

#     accuracy                           1.00     51781
#    macro avg       1.00      1.00      1.00     51781
# weighted avg       1.00      1.00      1.00     51781
print(confusion_matrix(y_test, y_pred))
# [[51268     2]
#  [    2   509]]
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
# Heading(mag): 0.1437407673420177
# Turn Rate: 0.1185320266676465
# Flight Path Angle - VV-[0]: 0.10821062115204479
# Baro Setting Pilot: 0.10193915453600468
# Ground Track - VV-[2]: 0.07623026321084747
# Transmission Chip Warning: 0.0633665947158324
# Transmission Oil Temp Warning: 0.04490843730821574
# Yaw Acceleration: 0.03749175846094667
# AP1 Status: 0.0333328903284939
# Ground Track Copilot: 0.032485752261187655
# Nav1 Ver Deviation: 0.030710511979500074
# Right Brake Pos: 0.028245172446840064
# NAV 2 DME Distance: 0.026887564983904436
# Yaw Rate: 0.02608253197107238
# NAV 2 DME Speed: 0.02599966347206947
# Flight Path Angle - VV-[1]: 0.02208851808661065
# Acceleration in Normal: 0.021367554052364103
# Tail Rotor Chip Warning: 0.019867776341259483
# TOGA Status: 0.0167248970251227
# Acceleration in Latitude: 0.012480979110518821
# Flight Path Angle - VV-[2]: 0.00930656454750032
