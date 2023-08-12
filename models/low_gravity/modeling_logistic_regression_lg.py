import pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, confusion_matrix, classification_report, make_scorer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from imblearn.under_sampling import RandomUnderSampler
from pathlib import Path
import warnings
from sklearn.exceptions import ConvergenceWarning

# Ignore specific warning categories
warnings.simplefilter("ignore", category=UserWarning)
warnings.simplefilter("ignore", category=ConvergenceWarning)
warnings.simplefilter("ignore", category=FutureWarning)

# this is the path to your pickle file (should be the same location as CSVs)
path = Path('../../data')

with open(f'{path}/low_g_pandas_2.0.2.pkl', 'rb') as file:
    df = pickle.load(file)

# drop index and create X and y
df = df.reset_index(drop=True)
# define independent variables and dependent variable
X = df.drop('LOW-G', axis=1)
y = df['LOW-G']

# create training set and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# create a pipeline
# pipe_with_rus = ImbPipeline([
#     ('scaler', StandardScaler()),
#     ('under', RandomUnderSampler(random_state=42)),
#     ('lr', LogisticRegression())
# ])

# pipe_without_rus = Pipeline([
#     ('scaler', StandardScaler()),
#     ('lr', LogisticRegression())
# ])

# params = [
#     {
#         'pipeline': [pipe_with_rus, pipe_without_rus],  # Both pipelines are tested
#         'pipeline__lr__penalty': ['l2', 'none'],
#         'pipeline__lr__C': [0.001, 0.01, 0.1, 1, 10, 100],
#         'pipeline__lr__class_weight': ['balanced'],
#         'pipeline__lr__solver': ['newton-cg', 'lbfgs', 'sag', 'saga'],
#         'pipeline__lr__fit_intercept': [True, False],
#         'pipeline__lr__random_state': [42],
#         'pipeline__lr__n_jobs': [-1],
#         'pipeline__lr__max_iter': [10000000]
#     },
#     {
#         'pipeline': [pipe_with_rus, pipe_without_rus],  # Both pipelines are tested
#         'pipeline__lr__penalty': ['l1'],
#         'pipeline__lr__C': [0.001, 0.01, 0.1, 1, 10, 100],
#         'pipeline__lr__class_weight': ['balanced'],
#         'pipeline__lr__solver': ['liblinear', 'saga'],
#         'pipeline__lr__fit_intercept': [True, False],
#         'pipeline__lr__random_state': [42],
#         'pipeline__lr__n_jobs': [-1],
#         'pipeline__lr__max_iter': [10000000]
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
#         'pipeline__lr__n_jobs': [-1],
#         'pipeline__lr__max_iter': [10000000]
#     }
# ]

# create pipeline
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('lr', LogisticRegression())
])

# parameters for the optimal model
params = {
    'lr__penalty': ['l1'],
    'lr__C': [1],
    'lr__class_weight': ['balanced'],
    'lr__solver': ['liblinear'],
    'lr__fit_intercept': [True, False],
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

#            0     1.0000    0.9858    0.9929     51437
#            1     0.3206    1.0000    0.4855       344

#     accuracy                         0.9859     51781
#    macro avg     0.6603    0.9929    0.7392     51781
# weighted avg     0.9955    0.9859    0.9895     51781

# confusion matrix for specific results
print(confusion_matrix(y_test, y_pred))
# [[50708   729]
#  [    0   344]]

# best hyperparameters to use for later analysis
print(best_params)
# {'lr__C': 1, 'lr__class_weight': 'balanced', 'lr__fit_intercept': True, 'lr__max_iter': 10000000, 'lr__n_jobs': -1, 'lr__penalty': 'l1', 'lr__random_state': 42, 'lr__solver': 'liblinear'}

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
# Airspeed(True): -210.9045
# Altitude(AGL): -186.3856
# Roll Acceleration: 120.0705
# Cyclic Pitch Pos-[0]: 104.8059
# Vert. Speed: -50.7661
# Rotor Torque-[0]: -27.4279
# Gross Weight: -27.1835
# Yaw Acceleration: -26.1127
# Collective Pos-[0]: -24.8406
# Altitude(MSL): -6.2643
# Roll: 6.0728
# Yaw: -4.4312
# Pitch: 3.8063
# Cyclic Roll Pos-[0]: 1.6252
# Rotor RPM-[0]: -1.5285
# Sideslip Angle: -1.1573
# Pitch Acceleration: 0.0000
# Pedal Pos: 0.0000
# Wind Speed(True): 0.0000
