import pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, confusion_matrix, classification_report, make_scorer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from imblearn.under_sampling import RandomUnderSampler
from pathlib import Path

# this is the path to your pickle file (should be the same location as CSVs)
path = Path('../data')

with open(path / 'dynamic_rollover.pkl', 'rb') as file:
    df = pickle.load(file)

# drop index and create X and y
df = df.reset_index(drop=True)
X = df.drop('Dynamic Rollover', axis=1)
y = df['Dynamic Rollover']

# create training set and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# separate pipelines, without undersampling is chosen and changed to 'pipe'
# pipe_with_rus = ImbPipeline([
#     ('scaler', StandardScaler()),
#     ('under', RandomUnderSampler(random_state=42)),
#     ('knn', KNeighborsClassifier())
# ])

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('knn', KNeighborsClassifier())
])

# separate parameter grids, without undersampling performed best and is changed to variable 'params'
# params_with_rus = {
#     'scaler__with_mean': [True],  # False
#     'knn__n_neighbors': [3], #  list(range(1, 11))
#     'knn__weights': ['distance'],  # 'uniform'
#     'knn__p': [1],  # 2
#     'knn__n_jobs': [-1]
# }

params = {
    'knn__n_neighbors': [3], #  list(range(1, 11))
    'knn__weights': ['distance'],  # 'uniform'
    'knn__p': [1],  # 2
    'knn__n_jobs': [-1]
}

# pipelines = [pipe_with_rus, pipe_without_rus]
# param_grids = [params_with_rus, params_without_rus]

# grid search with cross-validation
f1_scorer = make_scorer(f1_score)
strat_k_fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# instantiate the grid search loading pipeline, parameters, k-fold, and scorer
grid_search = GridSearchCV(estimator=pipe, param_grid=params, cv=strat_k_fold, scoring=f1_scorer)
# fit the grid search
grid_search.fit(X_train, y_train)

# save the best parameters from the grid search and the model with those parameters
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

# iterate through pipelines and param_grids, commented out as the best configuration is being used
# best_score = -1
# for pipe, params in zip(pipelines, param_grids):
#     grid_search = GridSearchCV(estimator=pipe, param_grid=params, cv=strat_k_fold, scoring=f1_scorer)
#     grid_search.fit(X_train, y_train)
    
#     if grid_search.best_score_ > best_score:
#         best_score = grid_search.best_score_
#         best_params = grid_search.best_params_
#         best_model = grid_search.best_estimator_

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

# showing best model configuration
print(best_model)
# Pipeline(steps=[('scaler', StandardScaler()),
#                 ('knn',
#                  KNeighborsClassifier(n_jobs=-1, n_neighbors=3, p=1,
#                                       weights='distance'))])

# best hyperparameters to use for later analysis
print(best_params)
# {'knn__n_jobs': -1, 'knn__n_neighbors': 3, 'knn__p': 1, 'knn__weights': 'distance'}
