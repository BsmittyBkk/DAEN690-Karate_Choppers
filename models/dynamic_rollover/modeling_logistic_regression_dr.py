import pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, confusion_matrix, classification_report, make_scorer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from imblearn.under_sampling import RandomUnderSampler
from sklearn.pipeline import FeatureUnion
from pathlib import Path
import warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.base import TransformerMixin, BaseEstimator

# Ignore specific warning categories
warnings.simplefilter("ignore", category=UserWarning)
warnings.simplefilter("ignore", category=ConvergenceWarning)
warnings.simplefilter("ignore", category=FutureWarning)

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

# # Define the pipelines
# pipe_with_rus = Pipeline([
#     ('scaler', StandardScaler()),
#     ('under', RandomUnderSampler(random_state=42)),
#     ('lr', LogisticRegression())
# ])

# pipe_without_rus = Pipeline([
#     ('scaler', StandardScaler()),
#     ('lr', LogisticRegression())
# ])

# # Setup the pipeline switcher
# pipeline_switcher = PipelineSwitcher(
#     estimator_choices={
#         'with_rus': pipe_with_rus,
#         'without_rus': pipe_without_rus
#     }
# )

# params = [
#     {
#         'pipeline': [pipe_with_rus, pipe_without_rus],  # Both pipelines are tested
#         'pipeline__lr__penalty': ['l2', 'none'],
#         'pipeline__lr__C': [0.001, 0.01, 0.1, 1, 10, 100],
#         'pipeline__lr__class_weight': ['balanced'],
#         'pipeline__lr__solver': ['newton-cg', 'lbfgs', 'sag', 'saga'],
#         'pipeline__lr__fit_intercept': [True, False],
#         'pipeline__lr__random_state': [42],
#         'pipeline__lr__n_jobs': [8],
#         'pipeline__lr__max_iter': [1000]
#     },
#     {
#         'pipeline': [pipe_with_rus, pipe_without_rus],  # Both pipelines are tested
#         'pipeline__lr__penalty': ['l1'],
#         'pipeline__lr__C': [0.001, 0.01, 0.1, 1, 10, 100],
#         'pipeline__lr__class_weight': ['balanced'],
#         'pipeline__lr__solver': ['liblinear', 'saga'],
#         'pipeline__lr__fit_intercept': [True, False],
#         'pipeline__lr__random_state': [42],
#         'pipeline__lr__n_jobs': [8],
#         'pipeline__lr__max_iter': [1000]
#     },
#     {
#         'pipeline': [pipe_with_rus, pipe_without_rus],  # Both pipelines are tested
#         'pipeline__lr__penalty': ['elasticnet'],
#         'pipeline__lr__C': [0.001, 0.01, 0.1, 1, 10, 100],
#         'pipeline__lr__class_weight': ['balanced'],
#         'pipeline__lr__solver': ['saga'],
#         'pipeline__lr__l1_ratio': [0, 0.25, 0.5, 0.75, 1],
#         'pipeline__lr__fit_intercept': [True, False],
#         'pipeline__lr__random_state': [42],
#         'pipeline__lr__n_jobs': [8],
#         'pipeline__lr__max_iter': [1000]
#     }
# ]

# create pipeline
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('lr', LogisticRegression())
])

# final parameters of the optimal model
params = {
    'lr__penalty': ['l2'],
    'lr__C': [1],
    'lr__class_weight': ['balanced'],
    'lr__solver': ['lbfgs'],
    'lr__fit_intercept': [True],
    'lr__random_state': [42],
    'lr__n_jobs': [-1],
    'lr__max_iter': [10000000]
}

# grid search with cross-validation
f1_scorer = make_scorer(f1_score)
strat_k_fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid_search = GridSearchCV(estimator=pipe, param_grid=params, cv=strat_k_fold, scoring=f1_scorer)
grid_search.fit(X_train, y_train)


# store the best hyperparameters and the best model
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_  

# use the best model to predict on the test set
y_pred = best_model.predict(X_test)

# print the classification report comparing the predicted values to the true values
print(classification_report(y_test, y_pred, digits=4))
#               precision    recall  f1-score   support

#            0     0.9994    0.8105    0.8951     51270
#            1     0.0476    0.9511    0.0907       511

#     accuracy                         0.8119     51781
#    macro avg     0.5235    0.8808    0.4929     51781
# weighted avg     0.9900    0.8119    0.8872     51781

# confusion matrix for specific results
print(confusion_matrix(y_test, y_pred))
# [[41556  9714]
#  [   25   486]]

# best hyperparameters to use for later analysis
print(best_params)
# {'lr__C': 1, 'lr__class_weight': 'balanced', 'lr__fit_intercept': True, 'lr__max_iter': 10000000, 'lr__n_jobs': -1, 'lr__penalty': 'l2', 'lr__random_state': 42, 'lr__solver': 'lbfgs'}

# accessing the coefficients of the logistic regression model
if 'rus' in best_model.named_steps:  # Check if the pipeline with RandomUnderSampling was chosen
    importances = best_model.named_steps['pipeline'].named_steps['lr'].coef_[0]
else:
    importances = best_model.named_steps['lr'].coef_[0]

# Retrieve feature names
feature_names = list(X_train.columns)

# Pair the features with their coefficients and sort them by the absolute value of the coefficients
sorted_features = sorted(zip(feature_names, importances), key=lambda x: abs(x[1]), reverse=True)

# Print the features and their coefficients
for feature, coeff in sorted_features:
    print(f"{feature}: {coeff:.4f}")
# Groundspeed: 25.2466
# Sideslip Angle: -24.5861
# Altitude(MSL): 11.7962
# Altitude(AGL): 10.0401
# Rotor Torque-[0]: 3.9528
# Pitch Acceleration: -2.1761
# Cyclic Pitch Pos-[0]: -1.7408
# Collective Pos-[0]: -1.7408
# Gross Weight: 1.4812
# Roll: 1.4136
# Pitch: -1.3045
# Yaw Acceleration: 1.2552
# Roll Acceleration: 1.1610
# Roll Rate: -0.5700
# Yaw Rate: -0.2781
# Pitch Rate: 0.1442
# Wind Speed(True): 0.1304
# Yaw: -0.0135
# Cyclic Roll Pos-[0]: -0.0000
# Pedal Pos: -0.0000
