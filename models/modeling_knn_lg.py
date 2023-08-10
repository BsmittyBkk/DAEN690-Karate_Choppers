import pickle
import pandas as pd
from sklearn.metrics import make_scorer, f1_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from pathlib import Path

# this is the path to your pickle file (should be the same location as CSVs)
path = Path('../data')

with open(path / 'low_g.pkl', 'rb') as file:
    df = pickle.load(file)

# drop index and create X and y
df = df.reset_index(drop=True)
X = df.drop('LOW-G', axis=1)
y = df['LOW-G']

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

# separate parameter grids
# params_with_rus = {
#     'scaler__with_mean': [True],  # False
#     'knn__n_neighbors': [3],  # list(range(1, 11))
#     'knn__weights': ['distance'],  # 'uniform'
#     'knn__p': [1],  # 2
#     'knn__n_jobs': [-1]
# }

params = {
    'knn__n_neighbors': [3],  # list(range(1, 11))
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

# use the best model to predict on the test set
y_pred = best_model.predict(X_test)

# print the classification report comparing the predicted values to the true values
print(classification_report(y_test, y_pred, digits=4))
#               precision    recall  f1-score   support

#            0     1.0000    1.0000    1.0000     51437
#            1     0.9971    1.0000    0.9985       344

#     accuracy                         1.0000     51781
#    macro avg     0.9986    1.0000    0.9993     51781
# weighted avg     1.0000    1.0000    1.0000     51781

# confusion matrix for specific results
print(confusion_matrix(y_test, y_pred))
# [[51436     1]
#  [    0   344]]

# showing best model configuration
print(best_model)
# Pipeline(steps=[('scaler', StandardScaler()),
#                 ('knn',
#                  KNeighborsClassifier(n_jobs=-1, n_neighbors=3, p=1,
#                                       weights='distance'))])

# best hyperparameters to use for later analysis
print(best_params)
# {'knn__n_jobs': -1, 'knn__n_neighbors': 3, 'knn__p': 1, 'knn__weights': 'distance'}
