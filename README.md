# DAEN690-Karate_Choppers
DAEN Summer Capstone Project focusing on   IDENTIFICATION, CHARACTERIZATION, AND ANALYSIS OF ROTORCRAFT FLIGHT MANEUVERS USING SIMULATED FLIGHT DATA

# Objectives
The goal of this project is to conduct reasearch, create a report, and create machine learning models to accurately and efficiently detect and classify a subset of maneuvers in rotocraft. The data used in this project is in-flight simulated data from a rotorcraft flight simulator using a AW-139 model. The selected subset of manuevers for this project are dynamic rollovers, retreating blade stalls, and low-g scenarios.

# Data
The data used in the project was limited to 2 file types, CSV's in the form of RoAR flight data, and PDF's of hand written flight logs by the the simulating pilot.

**CSV's**
The CSV's provided had instrument data recorded during the time of flight simulation, along with the respective timestamps. In total there were 2 different formats of the CSV, one containing 108 variables, and the other containing 201 variables. There is a mixture of datatypes including numerical, strings, and boolean. 

**PDF's**
The PDF's that were given were handwritten logs filled out by the pilot at the time of simulation. These logs had time stamps associated with the manuever/event that the pilot was conducting. The logs served as a resource for labeling events and manuevers of interest.

**STORAGE**
The data was originally going to be stored here in the repository, however due to file size and the limitation of available storage space in a free GitHub environment, the data will not reside in the Repo. The client has provided the team virtual cloud storage via AWS, which is where future data analysis and machine learning will be done. The AWS available to the team **is not** a limiting factor.

# Pre-Processing
Pre-processing thus far has centered around data exploration. Some things that have been found were variables with zero variance, correlations amongst variables, as well as data distribution visaulized through histograms. The team is looking to efficiently label the data and ingest new simulated data as provided by the client. 

# Algorithms

# Findings

# Conclusions

# Libraries Used
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
