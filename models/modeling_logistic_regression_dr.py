import pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import recall_score, confusion_matrix, classification_report, make_scorer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from pathlib import Path
import warnings
from sklearn.exceptions import ConvergenceWarning

# Ignore specific warning categories
warnings.simplefilter("ignore", category=UserWarning)
warnings.simplefilter("ignore", category=ConvergenceWarning)
warnings.simplefilter("ignore", category=FutureWarning)

# this is the path to your pickle file (should be the same location as CSVs)
path = Path('../../CSV')

with open(path / 'dynamic_rollover.pkl', 'rb') as file:
    df = pickle.load(file)

# drop index and create X and y
df = df.reset_index(drop=True)
# define independent variables and dependent variable
X = df.drop('Dynamic Rollover', axis=1)
y = df['Dynamic Rollover']

# create training set and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# parameter grid for finding the best hyperparameters
# params = [
#     {
#         'lr__penalty': ['l1', 'l2', 'none'],
#         'lr__C': [0.001, 0.01, 0.1, 1, 10, 100],
#         'lr__class_weight': ['balanced'],
#         'lr__solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
#         'lr__fit_intercept': [True, False],
#         'lr__random_state': [42],
#         'lr__n_jobs': [-1],
#         'lr__max_iter': [1000]
#     },
#     {
#         'lr__penalty': ['elasticnet'],
#         'lr__C': [0.001, 0.01, 0.1, 1, 10, 100],
#         'lr__class_weight': ['balanced'],
#         'lr__solver': ['saga'],
#         'lr__l1_ratio': [0, 0.25, 0.5, 0.75, 1],
#         'lr__fit_intercept': [True, False],
#         'lr__random_state': [42],
#         'lr__n_jobs': [-1],
#         'lr__max_iter': [1000]
#     }
# ]
params = [
    {
        'lr__penalty': ['l1', 'l2'],
        'lr__C': [0.01, 0.1, 1],
        'lr__class_weight': ['balanced'],
        'lr__solver': ['saga', 'lbfgs'], 
        'lr__fit_intercept': [True, False],
        'lr__random_state': [42],
        'lr__max_iter': [1000]
    },
    {
        'lr__penalty': ['elasticnet'],
        'lr__C': [0.01, 0.1, 1],
        'lr__class_weight': ['balanced'],
        'lr__solver': ['saga'],
        'lr__l1_ratio': [0.25, 0.5, 0.75],
        'lr__fit_intercept': [True, False],
        'lr__random_state': [42],
        'lr__n_jobs': [-1],
        'lr__max_iter': [1000]
    }
]


# create a pipeline
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('lr', LogisticRegression())
])

# grid search with cross-validation
recall_scorer = make_scorer(recall_score)
strat_k_fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid_search = GridSearchCV(estimator=pipe, param_grid=params, cv=strat_k_fold, scoring=recall_scorer)
grid_search.fit(X_train, y_train)

# store the best hyperparameters and the best model
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_  

# use the best model to predict on the test set
y_pred = best_model.predict(X_test)

# print the classification report comparing the predicted values to the true values
print(classification_report(y_test, y_pred, digits=4))

# confusion matrix for specific results
print(confusion_matrix(y_test, y_pred))

# best hyperparameters to use for later analysis
print(best_params)

# accessing the coefficients of the logistic regression model
importances = best_model.named_steps['lr'].coef_[0]

# Retrieve feature names
feature_names = list(X_train.columns)

# Pair the features with their coefficients and sort them by the absolute value of the coefficients
sorted_features = sorted(zip(feature_names, importances), key=lambda x: abs(x[1]), reverse=True)

# Print the features and their coefficients
for feature, coeff in sorted_features:
    print(f"{feature}: {coeff:.4f}")
