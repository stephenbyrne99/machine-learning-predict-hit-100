# Spotify Hot 100 Billboard Predictions
This is the repository for module **CSU44061-A-SEM101-202021: MACHINE LEARNING**. 
We decided to explore if we could build a model that could predict if a song would appear in the billboard hot 100. 

This was achieved by using features gathered from Spotify’s API such as:

- acousticness
- danceability
- energy
- instrumentalness 
- key
- liveness
- loudness
- mode
- speechiness
- tempp
- valence
- artist
 
This was mainly numerical data (with some categorical data) which we scaled during preprocessing, except for the artists name which was encoded as another category. 

We then used a variety of classifiers studied over the course such as a logistc w/l1 penalty + logistic  classifier, kNN and random forest classifier, and experimented with new methods like XGBoost, to output a binary decision of if a track would appear in the top 100 or not. 


## :ledger: Index

- [About](#beginner-about)
- [Usage](#zap-usage)
- [Installation](#electric_plug-installation)
- [File Structure](#file_folder-file-structure)
- [Branches](#cactus-branches)

##  :beginner: About / Report

Currently there are models built for kNN, random forest, logistic regression, logistic regression with an l1 penalty and XGBoost. The XGBoost and randomforest for dataset 2 were not included in the report due to space limitations.

There are two datasets, dataset 1 containing mainly negative non hits with spotify features, and dataset 2 which is more balanced between hits / non-hits and also contains additional features that are generated from the complexity of lyrics.

## :zap: Usage

This project was written only tested on Python 3.8.3. Make sure you have the correct libaries installed. 
You can do this using [pip](https://pypi.org/project/pip/).

Run the desired python file corresponing to what you would like to do.

###  :electric_plug: Installation

Required python libraries [modelling]:
- numpy == 1.18.5
- pandas == 1.0.5
- xgboost == 1.3.0.post0
- scikit_learn == 0.23.1

###  :file_folder: File Structure
Add a file structure here with the basic details about files, below is an example.

```
.
├── cleaning
│   ├── app.py
│   ├── clean.py
│   ├── h5out-cleaned.csv
│   ├── h5out.csv
│   ├── h5out-tagged.csv
│   └── tagging.py
├── featured_taggedv_2.1.csv
├── models
│   ├── basic-models.py
│   └── cv-models.py
├── README.md
├── scraping
│   ├── audiio.csv
│   ├── h5manipulation.py
│   ├── hdf5_getters.py
│   ├── hot_stuff_2.csv
│   ├── models.ipynb
│   ├── __pycache__
│   │   ├── hdf5_getters.cpython-36.pyc
│   │   └── hdf5_getters.cpython-38.pyc
│   ├── spotify_query.py
│   └── spotify_register.py
├── spotifytaste.ipynb
├── testmodels.ipynb
└── xgboost
    ├── xboost_tree.pdf
    └── xgbooster.py
```

 ### :cactus: Branches

**Steps to work with feature branch**

1. To start working on a new feature, create a new branch prefixed with `feat` and followed by feature name. (ie. `feat-FEATURE-NAME`)
2. Once you are done with your changes, you can raise PR.

g

1. Make a PR to `main` branch.
2. Comply with the best practices and guidelines e.g. where the PR concerns visual elements it should have an image showing the effect.
3. It must pass all continuous integration checks and get positive reviews.

After this, changes will be merged.
