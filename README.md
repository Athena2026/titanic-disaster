# Titanic Disaster Prediction

This repository contains a project to predict survivability of passengers on the Titanic using both Python and R. The project uses logistic regression and demonstrates Dockerized workflows for both languages.

# Project Structure

  titanic-disaster/

    src/
        app/            # Python container        
            main.py           
            Dockerfile    
            requirements.txt 
            
        r_app/          # R container       
            main.R
            Dockerfile
            install_packages.R
            
        data/           # Shared datasets
            train.csv
            test.csv
            gender_submission.csv
            
    .gitignore

    CODEOWNERS

# Getting Started
## 1. Clone the repository
git clone https://github.com/YourUsername/titanic-disaster.git
cd titanic-disaster

## 2. Prepare the data
Download the Titanic datasets from Kaggle Titanic Dataset  and place the files inside src/data/:

    train.csv

    test.csv

    gender_submission.csv
  

## 3. Running the Python Container

Navigate to the Python container folder: `cd src/app`


Build the Docker image: `docker build -t titanic-python:latest .`


Run the Docker container (mount the shared data folder):` docker run --rm -v "$(pwd)/../data:/app/src/data" titanic-python:latest`


### You should see outputs including:
```

  Data loading confirmation

  Missing value handling

  Feature engineering

  Training and validation accuracy

  Test predictions and comparison with sample submission
```


## 4. Running the R Container

Navigate to the R container folder: `cd src/r_app`

Build the Docker image: `docker build -t titanic-r:latest .`

Run the Docker container (mount the shared data folder): `docker run --rm -v "$(pwd)/../data:/r_app/src/data" titanic-r:latest`


### You should see outputs including:

```
Data loading confirmation

Missing value handling

Feature engineering

Training and validation accuracy

Test predictions and comparison with sample submission
```

## 5. Notes

Both containers share the same src/data folder.

Python container uses main.py with logistic regression from sklearn.

R container uses main.R with logistic regression using caret.

Ensure Docker is installed and running on your machine.

The .gitignore file excludes temporary files and the data folder for version control.

# References

Titanic Dataset: [Kaggle](https://www.kaggle.com/competitions/titanic/code)

Python packages: pandas, scikit-learn

R packages: dplyr, readr, ggplot2, caret
