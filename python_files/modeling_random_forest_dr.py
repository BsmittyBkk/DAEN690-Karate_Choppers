import pickle
import pandas as pd
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
    'rf__n_estimators': [200],  # 100, 200
    'rf__max_depth': [None],  # 5, 10
    'rf__min_samples_split': [2],  # 5, 10
    'rf__min_samples_leaf': [1],  # 2, 4
    'rf__max_features': ['log2'],  # 'sqrt'
    'rf__bootstrap': [False],  # False
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

print(classification_report(y_test, y_pred, digits=4))
#               precision    recall  f1-score   support

#            0     0.9999    1.0000    1.0000     51270
#            1     0.9980    0.9941    0.9961       511

#     accuracy                         0.9999     51781
#    macro avg     0.9990    0.9971    0.9980     51781
# weighted avg     0.9999    0.9999    0.9999     51781
print(confusion_matrix(y_test, y_pred))
# [[51269     1]
#  [    3   508]]
print(best_params)
# {'rf__bootstrap': False, 'rf__class_weight': 'balanced', 'rf__max_depth': None, 'rf__max_features': 'log2', 'rf__min_samples_leaf': 1, 'rf__min_samples_split': 2, 'rf__n_estimators': 200, 'rf__n_jobs': -1, 'rf__random_state': 42}
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
# Altitude(MSL): 0.13677521857315977
# Roll: 0.1259256041542172
# Yaw Acceleration: 0.12384875223219956
# Altitude(AGL): 0.12149699561681251
# Roll Rate: 0.08763573591649475
# Roll Acceleration: 0.04288178644714054
# Collective Pos-[0]: 0.04128339416698641
# Pitch Acceleration: 0.039931343314246144
# Rotor Torque-[0]: 0.03772277413378339
# Pitch Rate: 0.0365694458366391
# Cyclic Pitch Pos-[0]: 0.03577941368027163
# Gross Weight: 0.03186505781217571
# Wind Speed(True): 0.029632153359450672
# Groundspeed: 0.02629547841914004
# Pitch: 0.025183260848163202
# Yaw: 0.024023224600683412
# Sideslip Angle: 0.022858665461341066
# Yaw Rate: 0.010291695427094783
# Pedal Pos: 0.0
# Cyclic Roll Pos-[0]: 0.0