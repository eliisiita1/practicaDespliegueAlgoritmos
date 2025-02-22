import argparse
import pandas as pd
from split import split_data
from preprocessing import preprocess_data
from model import train_and_evaluate

def main(csv_path, test_size, random_state):
    """Ejecuta el pipeline completo: Split, Preprocesamiento y Entrenamiento."""

    print("📥 Dividiendo los datos en train y test...")
    X_train, X_test, y_train, y_test = split_data(csv_path, test_size)

    print("🔍 Preprocesando los datos de entrenamiento...")
    X_train = preprocess_data(X_train)

    print("🔍 Preprocesando los datos de prueba...")
    X_test = preprocess_data(X_test)

    print("🚀 Entrenando el modelo...")
    train_and_evaluate(X_train, X_test, y_train, y_test)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrenar modelo Titanic con MLflow")

    parser.add_argument("csv_path", type=str, help="Ruta del archivo CSV")
    parser.add_argument("--test_size", type=float, default=0.2, help="Tamaño del conjunto de prueba (default: 0.2)")
    parser.add_argument("--random_state", type=int, default=43, help="Random state para LogisticRegression (default: 43)")

    args = parser.parse_args()
    main(args.csv_path, args.test_size, args.random_state)
