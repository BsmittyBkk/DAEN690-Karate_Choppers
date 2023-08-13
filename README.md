# DAEN690-Karate_Choppers
DAEN Summer Capstone Project focusing on identification, characterization, and analysis of rotorcraft flight maneuvers using simulated flight data.

---

### Overview

This repository consists of the comprehensive analysis and modeling for two target variables: Dynamic Rollover and Low Gravity. The goal of this project is to conduct research, create a report, and create machine learning models to accurately and efficiently detect and classify a subset of maneuvers in rotorcraft. The data used in this project is in-flight simulated data from a rotorcraft flight simulator using a AW-139 model. The selected subset of maneuvers for this project are dynamic rollovers and low-g scenarios.

The main purpose of this README is to guide you through the structure and necessary steps to view the final results. If you wish to see the results immediately, you only need to ensure that all the requirements (found in `requirements.txt`) are installed, particularly pandas version 2.0.2, and then run the two final results Jupyter notebooks:

Dynamic Rollover: [final_results_dr.ipynb](final_results_dr.ipynb)

Low Gravity: [final_results_lg.ipynb](final_results_lg.ipynb)


### Repository Structure:
```
.
├── data
│   ├── dynamic_rollover_pandas_2.0.2.pkl
│   └── low_g_pandas_2.0.2.pkl
├── models
│   ├── dynamic_rollover
│   │   ├── modeling_decision_tree_dr.py
│   │   ├── modeling_knn_dr.py
│   │   ├── modeling_logistic_regression_dr.py
│   │   ├── modeling_random_forest_dr.py
│   │   ├── modeling_svm_dr.py
│   ├── low_gravity
│   │   ├── (similar structure as dynamic_rollover)
│   ├── preprocess_datasets.py
│   └── run_all_models.sh
├── notebooks
│   ├── dynamic_rollover
│   │   ├── visualization_decision_tree_dr.ipynb
│   │   ├── visualization_knn_dr.ipynb
│   │   ├── visualization_logistic_regression_dr.ipynb
│   │   ├── visualization_random_forest_dr.ipynb
│   │   ├── visualization_svm_dr.ipynb
│   │   └── pca_analysis_dr.ipynb
│   ├── low_gravity
│   │   ├── (similar structure as dynamic_rollover)
│   └── exploratory_data_analysis.ipynb
├── output
│   ├── final_results_dr.html
│   └── final_results_lg.html
├── .gitignore
├── final_results_dr.ipynb
├── final_results_lg.ipynb
├── requirements.txt
└── README.md
```


### Quick Start:

1. **Prerequisites**: Ensure all requirements are installed using:

    ```bash
    pip install -r requirements.txt
    ```

    Ensure you have `pandas` version `2.0.2` installed. This is crucial since the data files are pickled with this version.

2. **Final Results**: For immediate results:
    - Open the two main Jupyter notebooks (`final_results_dr.ipynb` and `final_results_lg.ipynb`).
    - Run them to view the final results for each target variable.

3. **HTML Outputs** (Optional based on feedback): If you don't want to run the Jupyter notebooks, you can view the provided HTML outputs for a snapshot of the final results:
    - `output/final_results_dr.html`
    - `output/final_results_lg.html`

### Detailed Breakdown:

- **data**: Contains datasets in pickle format. These files are tailored for `pandas 2.0.2`. Any other version might produce unpredictable results or cause an `_unpickle` error.

- **models**: This directory holds the raw python scripts used for model development, training, tuning, and analysis. We've tested multiple models such as decision trees, kNN, logistic regression, random forests, and SVM for binary classification. To run all models at once and retrieve the classification report and confusion matrix from each, you can use the shell script `run_all_models.sh`. Each model is set to the optimal hyperparameters, but the hyperparameter tuning code is commented out for reference.

- **notebooks**: Contains Jupyter notebooks detailing the analysis and visualizations for each model. Also includes a notebook for PCA analysis and an EDA (Exploratory Data Analysis) notebook.

### Notes:

- Data Source: The standalone python file in the `models` directory processes original CSVs from a folder (`CSV`) expected to be in a sibling directory. CSVs dated from 6/15/23 are processed and labeled using information derived from manual flightlog PDF analysis.

### Libraries used extensively in this project:

**Pandas**

Pandas was used primarily to begin pre-processing and exploring the data.
May access the most recent version to download below.

[Download available here](https://packaging.python.org/en/latest/tutorials/installing-packages/)

**Matplotlib**

Matplotlib was used to create some visaulizations in the pre-processing phase and in the analysis phase.
May access the most recent version to download below.

[Download available here](https://matplotlib.org/stable/users/installing/index.html)

**Seaborn**

Seaborn was used to create visualizations during the data exploration phase.
May access the most recent version to download below.

[Download available here](https://seaborn.pydata.org/installing.html)

**Numpy**

Numpy was used for its efficient numerical calculations in Python
May access the most recent version to download below.

[Download available here](https://numpy.org/install/)

**Scikit-learn**

Scikit-learn was used for pre-processing and future ML work.
May access the most recent version to download below.

[Download available here](https://scikit-learn.org/stable/install.html)

**Imbalanced-learn (imblearn)**

imbalanced-learn, often referred to as `imblearn`, was employed to handle imbalanced datasets by providing numerous resampling techniques.
You can access the most recent version to download below.

[Download available here](https://imbalanced-learn.org/stable/install.html)