import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, confusion_matrix, classification_report, make_scorer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from pathlib import Path

# this is the path to your pickle file (should be the same location as CSVs)
path = Path('../../data')

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
    'dt__max_depth': [None],  # 3, 5, 7, 10, 20
    'dt__min_samples_split': [2],  # 5, 10, 20, 30
    'dt__criterion': ['entropy'],  # 'gini'
    'dt__splitter': ['random'],  # 'best'
    'dt__max_features': ['sqrt'],  # None, 'log2'
    'dt__class_weight': ['balanced'],
    'dt__random_state': [42]
}

# create a pipeline to test the model
pipe = Pipeline([
    ('dt', DecisionTreeClassifier())
])

# create fl_scorer to use in the grid search
f1_scorer = make_scorer(f1_score)

# Grid search with cross-validation
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

#            0     1.0000    1.0000    1.0000     51437
#            1     0.9971    0.9971    0.9971       344

#     accuracy                         1.0000     51781
#    macro avg     0.9985    0.9985    0.9985     51781
# weighted avg     1.0000    1.0000    1.0000     51781
print(confusion_matrix(y_test, y_pred))
# [[51436     1]
#  [    1   343]]
print(best_params)
# {'dt__class_weight': 'balanced', 'dt__criterion': 'entropy', 'dt__max_depth': None, 'dt__max_features': 'sqrt', 'dt__min_samples_split': 2, 'dt__random_state': 42, 'dt__splitter': 'random'}

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
# Altitude(AGL): 0.34487532201489435
# Gross Weight: 0.26709247415574444
# Airspeed(True): 0.19369205527143554
# Yaw Acceleration: 0.11712940467812076
# Cyclic Roll Pos-[0]: 0.019578572326180092
# Pitch: 0.016637276745362247
# Pitch Acceleration: 0.010655728808544832
# Rotor Torque-[0]: 0.0067963708058225375
# Altitude(MSL): 0.006296761957752638
# Yaw: 0.0056088612522706485
# Cyclic Pitch Pos-[0]: 0.003624069605732644
# Roll Acceleration: 0.0028568697069675754
# Rotor RPM-[0]: 0.0021087991263167285
# Sideslip Angle: 0.0013995590251293121
# Vert. Speed: 0.0007102852334054477
# Collective Pos-[0]: 0.0005830592931449449
# Roll: 0.0003545299931751403
# Pedal Pos: 0.0
# Wind Speed(True): 0.0
