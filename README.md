# Predictive Modeling Using Machine Learning

## Project overview
This beginner-friendly supervised machine learning project predicts the species of an Iris flower using measurements. It trains and evaluates a Random Forest classifier and creates a confusion-matrix image.

## Files
- `main.py` — trains the model, prints evaluation metrics, saves the confusion matrix and a trained model.
- `iris.csv` — sample dataset included with the project.
- `requirements.txt` — required Python libraries.
- `outputs/` — generated charts and model file after running.

## Run on macOS
1. Extract the ZIP file.
2. Open Terminal and move into the extracted folder, for example:
   `cd ~/Downloads/predictive_modeling_ml_project`
3. (Recommended) Create a virtual environment:
   `python3 -m venv .venv`
   `source .venv/bin/activate`
4. Install libraries:
   `python3 -m pip install -r requirements.txt`
5. Run:
   `python3 main.py`

## What it does
- Loads the included Iris dataset from `iris.csv`
- Splits the data into training and test sets
- Trains a Random Forest classifier
- Prints accuracy, classification report, and confusion matrix
- Saves `outputs/confusion_matrix.png` and `outputs/iris_random_forest.joblib`

## Note
This is an educational demonstration. Reported accuracy is based on the included sample dataset and one train/test split; it is not a guarantee of performance on other data.
