# Función que procesa el dataset.
import pandas as pd

def preprocess_data(X):
    """Realiza el preprocesamiento del conjunto de datos Titanic."""

    # Eliminar columnas irrelevantes
    cols_to_drop = ['PassengerId', 'Name', 'Ticket', 'Cabin']
    X.drop(columns=cols_to_drop, inplace=True, errors='ignore')

    # Manejo de valores nulos
    X['Age'].fillna(X['Age'].median(), inplace=True)
    X['Embarked'].fillna(X['Embarked'].mode()[0], inplace=True)

    # Convertir "Sex" a binario (0 = male, 1 = female)
    X['Sex'] = X['Sex'].map({'male': 0, 'female': 1})

    # One-Hot Encoding para "Embarked"
    X = pd.get_dummies(X, columns=['Embarked'], drop_first=True)

    # Crear nueva columna 'AgeGroup'
    bins = [0, 18, 25, 65, float('inf')]
    labels = ['Child', 'Young', 'Adult', 'Senior']
    X['AgeGroup'] = pd.cut(X['Age'], bins=bins, labels=labels)

    # Convertir categorías de 'AgeGroup' en variables dummy
    X['Age_Child'] = (X['AgeGroup'] == 'Child').astype(int)
    X['Age_Young'] = (X['AgeGroup'] == 'Young').astype(int)
    X['Age_Adult'] = (X['AgeGroup'] == 'Adult').astype(int)
    X['Age_Senior'] = (X['AgeGroup'] == 'Senior').astype(int)

    X.drop(columns=['AgeGroup', 'Age'], inplace=True)

    # Crear columna 'HasFamily' y eliminar 'SibSp' y 'Parch'
    X['HasFamily'] = (X['SibSp'] + X['Parch'] > 0).astype(int)
    X.drop(columns=['SibSp', 'Parch'], inplace=True)

    return X
