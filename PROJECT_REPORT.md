# Project Report: Predictive Modeling Using Machine Learning

## Objective
To build a supervised machine learning model that predicts the species of an Iris flower from four measurements.

## Tools and libraries
Python, Pandas, Matplotlib, scikit-learn, and Joblib.

## Methodology
1. Load the included Iris dataset.
2. Separate flower measurements (features) from the species label (target).
3. Split the dataset into training data and testing data.
4. Train a Random Forest classification model.
5. Predict labels for the test data.
6. Evaluate performance with accuracy, a classification report, and a confusion matrix.
7. Save the confusion matrix image and trained model.

## Expected output
When `main.py` runs successfully, it prints the dataset size, train/test sizes, accuracy, precision/recall/F1-score, and confusion matrix. It saves a confusion matrix chart and a trained model in the `outputs` folder.

## Conclusion
This project demonstrates the main supervised-learning workflow: preparing data, splitting it into training and test sets, fitting a classifier, making predictions, and evaluating the results. The model's score depends on the dataset and split, so it should be reported from the actual program output.
