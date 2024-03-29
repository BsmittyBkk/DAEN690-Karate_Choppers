#!/bin/bash

# Make the script exit if any command in it fails
set -e

# Run Random Forest
echo "Running Random Forest Model..."
echo "Dyanmic Rollover: "
python dynamic_rollover/modeling_random_forest_dr.py
echo "Low-G: "
python low_gravity/modeling_random_forest_lg.py

# Run Decision Tree
echo "Running Decision Tree Model..."
echo "Dyanmic Rollover: "
python dynamic_rollover/modeling_decision_tree_dr.py
echo "Low-G: "
python low_gravity/modeling_decision_tree_lg.py

# Run SVM
echo "Running SVM Model..."
echo "Dyanmic Rollover: "
python dynamic_rollover/modeling_svm_dr.py
echo "Low-G: "
python low_gravity/modeling_svm_lg.py

# Run Logistic Regression
echo "Running Logistic Regression Model..."
echo "Dyanmic Rollover: "
python dynamic_rollover/modeling_logistic_regression_dr.py
echo "Low-G: "
python low_gravity/modeling_logistic_regression_lg.py

# Run KNN
echo "Running KNN Model..."
echo "Dyanmic Rollover: "
python dynamic_rollover/modeling_knn_dr.py
echo "Low-G: "
python low_gravity/modeling_knn_lg.py

echo "All models executed successfully!"
