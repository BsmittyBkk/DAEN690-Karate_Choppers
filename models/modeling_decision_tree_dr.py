import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, classification_report, make_scorer, f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from pathlib import Path

# this is the path to your pickle file (should be the same location as CSVs)
path = Path('../data')

with open(path / 'data/dynamic_rollover_pandas_2.0.2.pkl', 'rb') as file:
    df = pickle.load(file)

# drop index
df = df.reset_index(drop=True)
# define independent variables and dependent variable
X = df.drop('Dynamic Rollover', axis=1)
y = df['Dynamic Rollover']

# create training set and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# parameter grid for finding the best hyperparameters, only using top performing hyperparameters others are commented out for reference
params = {
    'dt__max_depth': [None],  # 3, 5, 7, 10, 20
    'dt__min_samples_split': [5],  # 2, 10, 20, 30
    'dt__criterion': ['entropy'],  # 'gini'
    'dt__splitter': ['best'],  # 'random'
    'dt__max_features': ['sqrt'],  # None, 'log2'
    'dt__class_weight': ['balanced'],
    'dt__random_state': [42]
}

# create a pipeline to test the model
pipe = Pipeline([
    ('dt', DecisionTreeClassifier())
])

# grid search with cross-validation
f1_scorer = make_scorer(f1_score)
strat_k_fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid_search = GridSearchCV(estimator=pipe, param_grid=params, cv=strat_k_fold, scoring=f1_scorer)
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
# {'dt__class_weight': 'balanced', 'dt__criterion': 'entropy', 'dt__max_depth': None, 'dt__max_features': 'sqrt', 'dt__min_samples_split': 5, 'dt__random_state': 42, 'dt__splitter': 'best'}

# access and sort feature importances
importances = best_model.named_steps['dt'].feature_importances_
sorted_indices = importances.argsort()[::-1]

# retrieve feature names
feature_names = list(X_train.columns)

# print the most important variables
print("Most important variables:")
for i in sorted_indices:
    print(f"{feature_names[i]}: {importances[i]}")
# Most important variables:
# Roll Rate: 0.34407440798516065
# Pitch Rate: 0.29253104023823845
# Cyclic Pitch Pos-[0]: 0.16410032565591165
# Sideslip Angle: 0.06380166226492942
# Altitude(MSL): 0.04295251515175139
# Yaw Rate: 0.03446784533764986
# Yaw: 0.02683330443285495
# Gross Weight: 0.014679462603123574
# Rotor Torque-[0]: 0.007486849921495837
# Roll Acceleration: 0.0030300838774952104
# Altitude(AGL): 0.002592234814762308
# Wind Speed(True): 0.002078542578341183
# Groundspeed: 0.0008362545739027731
# Roll: 0.0005013700590513865
# Yaw Acceleration: 1.9225659118450045e-05
# Collective Pos-[0]: 1.4874846213011887e-05
# Cyclic Roll Pos-[0]: 0.0
# Pedal Pos: 0.0
# Pitch: 0.0
# Pitch Acceleration: 0.0
