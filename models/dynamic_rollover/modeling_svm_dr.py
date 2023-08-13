import pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, confusion_matrix, classification_report, make_scorer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from imblearn.under_sampling import RandomUnderSampler, NearMiss
from imblearn.over_sampling import SMOTE, RandomOverSampler
from pathlib import Path

# this is the path to your pickle file (should be the same location as CSVs)
path = Path('../../data')

with open(path / 'data/dynamic_rollover_pandas_2.0.2.pkl', 'rb') as file:
    df = pickle.load(file)

# drop index and create X and y
df = df.reset_index(drop=True)
# define independent variables and dependent variable
X = df.drop('Dynamic Rollover', axis=1)
y = df['Dynamic Rollover']

# create training set and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# pipelines for finding the best hyperparameters, only using top performing pipeline with top performing parameters others are commented out for reference
# pipelines = {
#     'pipe_with_rus': ImbPipeline([
#         ('scaler', StandardScaler()),
#         ('under', RandomUnderSampler(random_state=42)),
#         ('svm', SVC())
#     ]),
#     'pipe_with_smote': ImbPipeline([
#         ('scaler', StandardScaler()),
#         ('smote', SMOTE(random_state=42)),
#         ('svm', SVC())
#     ]),
#     'pipe_with_ros': ImbPipeline([
#         ('scaler', StandardScaler()),
#         ('ros', RandomOverSampler(random_state=42)),
#         ('svm', SVC())
#     ]),
#     'pipe_with_nearmiss': ImbPipeline([
#         ('scaler', StandardScaler()),
#         ('nearmiss', NearMiss(version=3)),
#         ('svm', SVC())
#     ]),
#     'pipe_without_resampling': Pipeline([
#         ('scaler', StandardScaler()),
#         ('svm', SVC())
#     ])
# }

# param_grids = {
#     'pipe_with_rus': {
#         'svm__C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
#         'svm__class_weight': ['balanced', None],
#         'svm__fit_intercept': [True, False],
#         'svm__random_state': [42],
#         'svm__max_iter': [1000]
#     },
#     'pipe_with_smote': {
#         'svm__C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
#         'svm__class_weight': ['balanced', None],
#         'svm__fit_intercept': [True, False],
#         'svm__random_state': [42],
#         'svm__max_iter': [1000]
#     },
#     'pipe_with_ros': {
#         'svm__C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
#         'svm__class_weight': ['balanced', None],
#         'svm__fit_intercept': [True, False],
#         'svm__random_state': [42],
#         'svm__max_iter': [1000]
#     },
#     'pipe_with_nearmiss': {
#         'svm__C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
#         'svm__class_weight': ['balanced', None],
#         'svm__fit_intercept': [True, False],
#         'svm__random_state': [42],
#         'svm__max_iter': [1000]
#     },
    # 'pipe_without_resampling': {
    #     'svm__C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
    #     'svm__class_weight': ['balanced', None],
    #     'svm__fit_intercept': [True, False],
    #     'svm__random_state': [42],
    #     'svm__max_iter': [1000]
    # }
# }

# create a pipeline
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC())
])

# parameter grid for finding the best hyperparameters
params = {
    'svm__gamma': [1],
    'svm__C': [1000],
    'svm__kernel': ['rbf'],
    'svm__class_weight': ['balanced'],
    'svm__random_state': [42]
}

# grid search with cross-validation
f1_scorer = make_scorer(f1_score)
strat_k_fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# instantiate the grid search loading pipeline, parameters, k-fold, and scorer
grid_search = GridSearchCV(estimator=pipe, param_grid=params, cv=strat_k_fold, scoring=f1_scorer)
# fit the grid search
grid_search.fit(X_train, y_train)

# below iterates through the various pipelines and parameter sets to find the optimal combination, commented out using only best configuration
# best_score = -1
# Iterate over pipelines and their corresponding parameter grids
# for pipe_name, pipe in pipelines.items():
#     grid_search = GridSearchCV(estimator=pipe, param_grid=params, cv=strat_k_fold, scoring=f1_scorer)
#     grid_search.fit(X_train, y_train)
    
#     if grid_search.best_score_ > best_score:
#         best_score = grid_search.best_score_
#         best_params = grid_search.best_params_
#         best_model = grid_search.best_estimator_

# store the best hyperparameters and the best model
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_  

# use the best model to predict on the test set
y_pred = best_model.predict(X_test)

# print the classification report comparing the predicted values to the true values
print(classification_report(y_test, y_pred, digits=4))
#               precision    recall  f1-score   support

#            0     0.9999    0.9992    0.9996     51270
#            1     0.9236    0.9941    0.9576       511

#     accuracy                         0.9991     51781
#    macro avg     0.9618    0.9967    0.9786     51781
# weighted avg     0.9992    0.9991    0.9991     51781

# confusion matrix for specific results
print(confusion_matrix(y_test, y_pred))
# [[51228    42]
#  [    3   508]]

# best hyperparameters to use for later analysis
print(best_params)
# {'svm__C': 1000, 'svm__class_weight': 'balanced', 'svm__gamma': 1, 'svm__kernel': 'rbf', 'svm__random_state': 42}
