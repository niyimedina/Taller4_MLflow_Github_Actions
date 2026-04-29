# src/train.py
import os
import sys
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from mlflow.models import infer_signature
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score


EXPERIMENT_NAME = "Taller4-MLflow-GitHubActions"
MODEL_NAME = "pima_diabetes_model"
DATA_PATH = os.path.join(os.getcwd(), "data", "pima-indians-diabetes.csv")


def main():
    print("Iniciando entrenamiento")

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

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    mlruns_dir = os.path.join(os.getcwd(), "mlruns")
    os.makedirs(mlruns_dir, exist_ok=True)

    #mlflow.set_tracking_uri(os.path.abspath(mlruns_dir))
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run():
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("dataset", "pima-indians-diabetes.csv")
        mlflow.log_param("target_column", target_column)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score", f1)

        signature = infer_signature(X_train, model.predict(X_train))
        input_example = X_train.head(5)

        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=signature,
            input_example=input_example
            #,registered_model_name=MODEL_NAME,
        )

        with open("model_uri.txt", "w", encoding="utf-8") as f:
            f.write(model_info.model_uri)

        model_path = os.path.join(os.getcwd(), "src", "model.pkl")
        joblib.dump(model, model_path)

        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"Modelo registrado en MLflow: {model_info.model_uri}")


if __name__ == "__main__":
    main()