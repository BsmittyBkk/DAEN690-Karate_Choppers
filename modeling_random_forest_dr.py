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
    'rf__min_samples_split': [5],  # 5, 10
    'rf__min_samples_leaf': [2],  # 2, 4
    'rf__max_features': ['log2'],  # 'sqrt'
    'rf__bootstrap': [True],  # False
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
# Pitch Acceleration: 0.17851542451386945
# Roll Acceleration: 0.16179783106270154
# Wind Direction(Mag): 0.12959880208467317
# Roll Rate: 0.1202688704160081
# Pitch Rate: 0.10789304892965978
# Yaw Acceleration: 0.08247524752210915
# Roll: 0.07344647020163995
# Gross Weight: 0.05481042394897525
# Groundspeed: 0.04017394841957257
# Wind Speed(True): 0.03749036364926704
# Fuel Weight: 0.013529569251524105

