# Función que divide el dataset original en train y test.
import pandas as pd
from sklearn.model_selection import train_test_split

def split_data(csv_path, test_size=0.2):
    """Carga el dataset y lo divide en entrenamiento y prueba."""
    
    # Cargar los datos
    df = pd.read_csv(csv_path)

    # Separar la variable objetivo
    X = df.drop(columns=['Survived'])
    y = df['Survived']

    # División en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)

    return X_train, X_test, y_train, y_test
