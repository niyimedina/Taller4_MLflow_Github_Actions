# src/validate.py
import os
import sys
import mlflow.pyfunc
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report

THRESHOLD_ACCURACY = 0.70
DATA_PATH = os.path.join(os.getcwd(), "data", "pima-indians-diabetes.csv")
MODEL_URI_PATH = os.path.join(os.getcwd(), "model_uri.txt")


def main():
    print("Iniciando validación del modelo")

    if not os.path.exists(DATA_PATH):
        print(f"No se encontró el dataset en: {DATA_PATH}")
        sys.exit(1)

    #df = pd.read_csv(DATA_PATH)
    columns = [
    "pregnancies", "glucose", "blood_pressure", "skin_thickness",
    "insulin", "bmi", "diabetes_pedigree", "age", "target"
    ]

    df = pd.read_csv(DATA_PATH, names=columns)

    target_column = "target" if "target" in df.columns else "outcome"

    X = df.drop(columns=[target_column])
    y = df[target_column]

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    if not os.path.exists(MODEL_URI_PATH):
        print("No se encontró model_uri.txt. Ejecuta primero make train.")
        sys.exit(1)

    with open(MODEL_URI_PATH, "r", encoding="utf-8") as f:
        model_uri = f.read().strip()

    print(f"Cargando modelo registrado desde MLflow: {model_uri}")
    model = mlflow.pyfunc.load_model(model_uri)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("Reporte de clasificación:")
    print(classification_report(y_test, y_pred))

    if accuracy >= THRESHOLD_ACCURACY:
        print("El modelo cumple los criterios de calidad.")
        sys.exit(0)
    else:
        print("El modelo no cumple el umbral mínimo.")
        sys.exit(1)


if __name__ == "__main__":
    main()