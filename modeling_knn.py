import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, make_scorer, f1_score
# from imblearn.over_sampling import SMOTE
# from sklearn.ensemble import RandomForestClassifier

# this is the path to your pickle file (should be the same location as CSVs)
path = r'../CSV'

# the below function verifies that the dataframe you are working with is the same shape as the anticipated dataframe
def test_dataframe_shape():
    # load the dataframe to be tested
    with open(f'{path}/working_df.pkl', 'rb') as file:
        df = pickle.load(file)
    # Perform the shape validation
    assert df.shape == (487470, 118)
    return df

# working dataframe that has 'Label', 'Dynamic Rollover', 'LOW-G' as the final 3 columns
df = test_dataframe_shape().reset_index(drop=True)

# # to test on Dynamic Rollover
# df = df.drop(columns=['Label', 'LOW-G'])
# # define X and y Dynamic Rollover
# X = df.drop('Dynamic Rollover', axis=1)
# y = df['Dynamic Rollover']

# to test on LOW-G
df = df.drop(columns=['Label', 'Dynamic Rollover'])
# define X and y for LOW-G
X = df.drop('LOW-G', axis=1)
y = df['LOW-G']

scaler = StandardScaler()

# create training set and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

knn = KNeighborsClassifier(n_neighbors=2, n_jobs=-1)
knn.fit(X_train_scaled, y_train)
y_pred = knn.predict(X_test_scaled)

# try oversampling the target variable
# smote = SMOTE(random_state=42)
# X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

# param_grid = {
#     'n_estimators': [100, 200, 300],
#     'max_depth': [None, 5, 10],
#     'min_samples_split': [2, 5, 10],
#     'min_samples_leaf': [1, 2, 4],
#     'max_features': ['auto', 'sqrt', 'log2'],
#     'bootstrap': [True, False]
# }


# train the random forest model on the resampled data
# rf = RandomForestClassifier(random_state=42)
# f1_scorer = make_scorer(f1_score)
# grid_search = GridSearchCV(
#     estimator=rf, param_grid=param_grid, cv=5, scoring=f1_scorer)
# grid_search.fit(X_train_resampled, y_train_resampled)

# best_params = grid_search.best_params_
# best_model = grid_search.best_estimator_  
# y_pred = best_model.predict(X_test_scaled)

# rf.fit(X_train_resampled, y_train_resampled)

# # predict on the test set
# y_pred = rf.predict(X_test_scaled)

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))