
# Función que entrena y evalúa el modelo.
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score

def train_and_evaluate(X_train, X_test, y_train, y_test):
    """Entrena y evalúa un modelo de Regresión Logística."""

    # Configurar MLflow
    mlflow.set_experiment("Titanic_Survival_Classification")

    with mlflow.start_run():
        # Definir y entrenar el modelo
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X_train, y_train)

        # Predicciones
        y_pred = model.predict(X_test)

        # Evaluación del modelo
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)

        # Registrar métricas en MLflow
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)

        # Guardar el modelo en MLflow
        mlflow.sklearn.log_model(model, "LogisticRegressionModel")

        print(f"✅ Accuracy: {accuracy:.4f}, Precisión: {precision:.4f}, Recall: {recall:.4f}")
