import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, make_scorer, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

# this is the path to your pickle file (should be the same location as CSVs)
path = r'../CSV'

# the below function verifies that the dataframe you are working with is the same shape as the anticipated dataframe
def test_dataframe_shape():
    # load the dataframe to be tested
    with open(f'{path}/working_df2.pkl', 'rb') as file:
        df = pickle.load(file)
    # Perform the shape validation
    # assert df.shape == (258905, 118)
    return df

# working dataframe that has 'Label', 'Dynamic Rollover', 'LOW-G' as the final 3 columns
df = test_dataframe_shape().reset_index(drop=True)

## to test on Low-G
df = df.drop(columns=['Label', 'Dynamic Rollover', 'Main Rotor Angle Slow', 'Swashplate Rotor 072', 'Swashplate Rotor 288'])
# Main Rotor Angle Slow: 0.0
# Swashplate Rotor 072: 0.0
# Swashplate Rotor 288: 0.0

# define X and y Dynamic Rollover
X = df.drop('LOW-G', axis=1)
y = df['LOW-G']

# create training set and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# parameter grid
params = {
    'rf__n_estimators': [50],  # 100, 200
    'rf__max_depth': [None],  # 5, 10
    'rf__min_samples_split': [2],  # 5, 10
    'rf__min_samples_leaf': [1],  # 2, 4
    'rf__max_features': ['sqrt'],  # 'log2'
    'rf__bootstrap': [True],  # False
    'rf__class_weight': ['balanced'],
    'rf__random_state': [42],
    'rf__n_jobs': [-1]
}

# create a pipeline to test the model
pipeline = Pipeline([
    ('rf', RandomForestClassifier())
])

# create fl_scorer to use in the grid search
f1_scorer = make_scorer(f1_score)

# set up k-fold cross validation
strat_k_fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# instantiate the grid search loading pipeline, parameters, k-fold, and scorer
grid_search = GridSearchCV(estimator=pipeline, param_grid=params, cv=strat_k_fold, scoring=f1_scorer)
# fit the grid search
grid_search.fit(X_train, y_train)

# save the best parameters from the grid search and the model with those parameters
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_  

# predict on the test set with the best model
y_pred = best_model.predict(X_test)

# print outputs
print(classification_report(y_test, y_pred))
#               precision    recall  f1-score   support

#            0       1.00      1.00      1.00     51437
#            1       1.00      1.00      1.00       344

#     accuracy                           1.00     51781
#    macro avg       1.00      1.00      1.00     51781
# weighted avg       1.00      1.00      1.00     51781
print(confusion_matrix(y_test, y_pred))
# [[51437     0]
#  [    1   343]]
print(best_params)
# 'rf__bootstrap': True, 'rf__class_weight': 'balanced', 'rf__max_depth': None, 'rf__max_features': 'sqrt', 'rf__min_samples_leaf': 1, 'rf__min_samples_split': 2, 'rf__n_estimators': 50, 'rf__n_jobs': -1, 'rf__random_state': 42

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
# Pitch Path: 0.182232561159997
# GPS 1 DME Speed: 0.1583920938049593
# NAV 1 DME Time: 0.1337096310497321
# NAV 1 DME Speed: 0.08218891633505629
# Ground Track - VV-[0]: 0.07649893204183678
# Vert. Speed: 0.07531295161043715
# Pedal Pos: 0.07077139101877601
# Heading(mag): 0.07058102562528272
# Induced Velo Behind Disc-[0]: 0.03502328369058118
# Yaw Acceleration: 0.03391873602206241
# Pitch Rate: 0.026999031988835458
# Swashplate Rotor 000: 0.01378417768732544
# Compass Heading: 0.010648522822412895
# Swashplate Rotor 216: 0.008519388517333191
# Swashplate Rotor 144: 0.006979004971880761
# Main Rotor Pos: 0.004851132719865073
# GPS 1 DME Distance: 0.0030321744370437657
# Roll Rate: 0.002180796217364862
# Roll Acceleration: 0.0016402372094620986
# NAV 2 NAV ID: 0.0008388676276857117
# Heading Path: 0.0007788362622645159
# Pitch Acceleration: 0.00042640108875176676
# Induced Velo Behind Disc-[1]: 0.0004021216712105926
# Flight Path Angle - VV-[0]: 0.00018044632564056035
# GPS Hor Deviation: 0.0001093380940175685
# Swashplate Phase: 1.8493400171409013e-13
