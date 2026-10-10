"""Predictive Modeling Using Machine Learning - beginner project."""
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

def main():
    data_path = BASE_DIR / "iris.csv"
    df = pd.read_csv(data_path)

    target_column = "species"
    X = df.drop(columns=[target_column])
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    print("\n=== Predictive Modeling Results ===")
    print(f"Dataset rows: {len(df)}")
    print(f"Training rows: {len(X_train)}")
    print(f"Testing rows: {len(X_test)}")
    print(f"Accuracy: {accuracy:.2%}")
    print("\nClassification Report:")
    print(classification_report(y_test, predictions, zero_division=0))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    display = ConfusionMatrixDisplay.from_estimator(
        model, X_test, y_test, cmap="Blues", xticks_rotation=30
    )
    display.ax_.set_title("Iris Species - Random Forest Confusion Matrix")
    plt.tight_layout()
    chart_path = OUTPUT_DIR / "confusion_matrix.png"
    plt.savefig(chart_path, dpi=150)
    plt.close()

    model_path = OUTPUT_DIR / "iris_random_forest.joblib"
    joblib.dump(model, model_path)
    print(f"\nSaved chart: {chart_path}")
    print(f"Saved trained model: {model_path}")

    # Example prediction using the four measurements in centimeters.
    sample = pd.DataFrame(
        [[5.1, 3.5, 1.4, 0.2]],
        columns=X.columns
    )
    print("\nExample prediction for [5.1, 3.5, 1.4, 0.2]:")
    print(model.predict(sample)[0])

if __name__ == "__main__":
    main()
