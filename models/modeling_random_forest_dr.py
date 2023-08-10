import pickle
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, make_scorer, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from pathlib import Path

# this is the path to your pickle file (should be the same location as CSVs)
path = Path('../data')

with open(f'{path}/dynamic_rollover.pkl', 'rb') as file:
    df = pickle.load(file)

# drop index
df = df.reset_index(drop=True)
# define independent variables and dependent variable
X = df.drop('Dynamic Rollover', axis=1)
y = df['Dynamic Rollover']

# create training set and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# parameter grid for finding the best hyperparameters
params = {
    'rf__n_estimators': [200],  # 100, 200
    'rf__max_depth': [None],  # 5, 10
    'rf__min_samples_split': [2],  # 5, 10
    'rf__min_samples_leaf': [1],  # 2, 4
    'rf__max_features': ['log2'],  # 'sqrt'
    'rf__bootstrap': [True],  # False
    'rf__class_weight': ['balanced'],
    'rf__random_state': [42],
    'rf__n_jobs': [-1]
}

# create a pipeline
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

#            0     0.9999    1.0000    1.0000     51270
#            1     0.9980    0.9941    0.9961       511

#     accuracy                         0.9999     51781
#    macro avg     0.9990    0.9971    0.9980     51781
# weighted avg     0.9999    0.9999    0.9999     51781

# confusion matrix for specific results
print(confusion_matrix(y_test, y_pred))
# [[51269     1]
#  [    3   508]]

# best hyperparameters to use for later analysis
print(best_params)
# {'rf__bootstrap': False, 'rf__class_weight': 'balanced', 'rf__max_depth': None, 'rf__max_features': 'log2', 'rf__min_samples_leaf': 1, 'rf__min_samples_split': 2, 'rf__n_estimators': 200, 'rf__n_jobs': -1, 'rf__random_state': 42}

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
