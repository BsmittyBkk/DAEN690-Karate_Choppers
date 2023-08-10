import pickle
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, make_scorer, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from pathlib import Path

# this is the path to your pickle file (should be the same location as CSVs)
path = Path('../data')

with open(f'{path}/low_g_pandas_2.0.2.pkl', 'rb') as file:
    df = pickle.load(file)

# drop index
df = df.reset_index(drop=True)
# define independent variables and dependent variable
X = df.drop('LOW-G', axis=1)
y = df['LOW-G']

# create training set and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# parameter grid for finding the best hyperparameters
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
pipe = Pipeline([
    ('rf', RandomForestClassifier())
])

# create f1_scorer to use in the grid search
f1_scorer = make_scorer(f1_score)
# set up k-fold cross validation
strat_k_fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# instantiate the grid search loading pipeline, parameters, k-fold, and scorer
grid_search = GridSearchCV(estimator=pipe, param_grid=params, cv=strat_k_fold, scoring=f1_scorer)
# fit the grid search
grid_search.fit(X_train, y_train)

# save the best parameters from the grid search and the model with those parameters
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_  

# predict on the test set with the best model
y_pred = best_model.predict(X_test)

# print the classification report comparing the predicted values to the true values
print(classification_report(y_test, y_pred, digits=4))
#               precision    recall  f1-score   support

#            0     1.0000    1.0000    1.0000     51437
#            1     1.0000    0.9971    0.9985       344

#     accuracy                         1.0000     51781
#    macro avg     1.0000    0.9985    0.9993     51781
# weighted avg     1.0000    1.0000    1.0000     51781

# confusion matrix for specific results
print(confusion_matrix(y_test, y_pred))
# [[51437     0]
#  [    1   343]]

# best hyperparameters to use for later analysis
print(best_params)
# {'rf__bootstrap': True, 'rf__class_weight': 'balanced', 'rf__max_depth': None, 'rf__max_features': 'log2', 'rf__min_samples_leaf': 1, 'rf__min_samples_split': 2, 'rf__n_estimators': 50, 'rf__n_jobs': -1, 'rf__random_state': 42}

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
# Altitude(MSL): 0.28447842882329966
# Yaw: 0.1467434258765614
# Yaw Acceleration: 0.13668113110557897
# Gross Weight: 0.08385280109191866
# Altitude(AGL): 0.07968419860708995
# Pitch Acceleration: 0.07864012393187736
# Vert. Speed: 0.0502623696731662
# Cyclic Pitch Pos-[0]: 0.04893398302304429
# Roll Acceleration: 0.045921583240967866
# Airspeed(True): 0.017247984244272055
# Rotor Torque-[0]: 0.010176746962487609
# Collective Pos-[0]: 0.007324957896244195
# Rotor RPM-[0]: 0.0039711342713507885
# Cyclic Roll Pos-[0]: 0.0020151047593346878
# Sideslip Angle: 0.0019747532238404424
# Roll: 0.0017344549753597071
# Pitch: 0.0003568182936061047
# Pedal Pos: 0.0
# Wind Speed(True): 0.0
