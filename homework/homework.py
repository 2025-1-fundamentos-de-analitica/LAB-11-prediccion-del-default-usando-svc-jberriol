# flake8: noqa: E501
#
# En este dataset se desea pronosticar el default (pago) del cliente el próximo
# mes a partir de 23 variables explicativas.
#
#   LIMIT_BAL: Monto del credito otorgado. Incluye el credito individual y el
#              credito familiar (suplementario).
#         SEX: Genero (1=male; 2=female).
#   EDUCATION: Educacion (0=N/A; 1=graduate school; 2=university; 3=high school; 4=others).
#    MARRIAGE: Estado civil (0=N/A; 1=married; 2=single; 3=others).
#         AGE: Edad (years).
#       PAY_0: Historia de pagos pasados. Estado del pago en septiembre, 2005.
#       PAY_2: Historia de pagos pasados. Estado del pago en agosto, 2005.
#       PAY_3: Historia de pagos pasados. Estado del pago en julio, 2005.
#       PAY_4: Historia de pagos pasados. Estado del pago en junio, 2005.
#       PAY_5: Historia de pagos pasados. Estado del pago en mayo, 2005.
#       PAY_6: Historia de pagos pasados. Estado del pago en abril, 2005.
#   BILL_AMT1: Historia de pagos pasados. Monto a pagar en septiembre, 2005.
#   BILL_AMT2: Historia de pagos pasados. Monto a pagar en agosto, 2005.
#   BILL_AMT3: Historia de pagos pasados. Monto a pagar en julio, 2005.
#   BILL_AMT4: Historia de pagos pasados. Monto a pagar en junio, 2005.
#   BILL_AMT5: Historia de pagos pasados. Monto a pagar en mayo, 2005.
#   BILL_AMT6: Historia de pagos pasados. Monto a pagar en abril, 2005.
#    PAY_AMT1: Historia de pagos pasados. Monto pagado en septiembre, 2005.
#    PAY_AMT2: Historia de pagos pasados. Monto pagado en agosto, 2005.
#    PAY_AMT3: Historia de pagos pasados. Monto pagado en julio, 2005.
#    PAY_AMT4: Historia de pagos pasados. Monto pagado en junio, 2005.
#    PAY_AMT5: Historia de pagos pasados. Monto pagado en mayo, 2005.
#    PAY_AMT6: Historia de pagos pasados. Monto pagado en abril, 2005.
#
# La variable "default payment next month" corresponde a la variable objetivo.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# clasificación están descritos a continuación.
#
#
# Paso 1.
# Realice la limpieza de los datasets:
# - Renombre la columna "default payment next month" a "default".
# - Remueva la columna "ID".
# - Elimine los registros con informacion no disponible.
# - Para la columna EDUCATION, valores > 4 indican niveles superiores
#   de educación, agrupe estos valores en la categoría "others".
# - Renombre la columna "default payment next month" a "default"
# - Remueva la columna "ID".
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Descompone la matriz de entrada usando PCA. El PCA usa todas las componentes.
# - Estandariza la matriz de entrada.
# - Selecciona las K columnas mas relevantes de la matrix de entrada.
# - Ajusta una maquina de vectores de soporte (svm).
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use la función de precision
# balanceada para medir la precisión del modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas de precision, precision balanceada, recall,
# y f1-score para los conjuntos de entrenamiento y prueba.
# Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# Este diccionario tiene un campo para indicar si es el conjunto
# de entrenamiento o prueba. Por ejemplo:
#
# {'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
# {'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}
#
#
# Paso 7.
# Calcule las matrices de confusion para los conjuntos de entrenamiento y
# prueba. Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'cm_matrix', 'dataset': 'train', 'true_0': {"predicted_0": 15562, "predicte_1": 666}, 'true_1': {"predicted_0": 3333, "predicted_1": 1444}}
# {'type': 'cm_matrix', 'dataset': 'test', 'true_0': {"predicted_0": 15562, "predicte_1": 650}, 'true_1': {"predicted_0": 2490, "predicted_1": 1420}}
#
import gzip
import json
import os
import pickle
import zipfile
from glob import glob

import pandas as pd  # type: ignore
from sklearn.compose import ColumnTransformer  # type: ignore
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif  # type: ignore
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV  # type: ignore
from sklearn.pipeline import Pipeline  # type: ignore
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # type: ignore
from sklearn.svm import SVC  # type: ignore

#Se cargan los datos del dataset
def load_dataset(input_folder):
    dataframes = []
    zip_files = glob(f"{input_folder}/*")
    
    for zip_path in zip_files:
        with zipfile.ZipFile(zip_path, mode="r") as zip_file:
            for filename in zip_file.namelist():
                with zip_file.open(filename) as file:
                    dataframes.append(pd.read_csv(file, sep=",", index_col=0))
    
    return dataframes

#Se crea la carpeta de los resultados
def create_output_folder(output_folder):
    if os.path.exists(output_folder):
        for file in glob(f"{output_folder}/*"):
            os.remove(file)
        os.rmdir(output_folder)
    os.makedirs(output_folder)

#Se realiza el preprocesamiento de los datos
def preprocess_dataframe(df):
    """Paso 1"""
    df = df.copy()
    df = df.rename(columns={"default payment next month": "default"})
    df = df.loc[df["MARRIAGE"] != 0]
    df = df.loc[df["EDUCATION"] != 0]
    df["EDUCATION"] = df["EDUCATION"].apply(lambda x: 4 if x >= 4 else x)
    return df.dropna()

#Parte del preprocesamiento de los datos
def split_features_target(df):
    """Paso 2"""
    return df.drop(columns=["default"]), df["default"]
# Se construye el pipeline del modelo de clasificación
def build_pipeline():
    """Paso 3"""
    categorical_features = ["SEX", "EDUCATION", "MARRIAGE"]
    numerical_features = [
        "LIMIT_BAL", "AGE", "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
        "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
        "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6",
    ]
    
    preprocessing = ColumnTransformer(
        [
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("scaling", StandardScaler(with_mean=True, with_std=True), numerical_features),
        ],
    )
    
    return Pipeline(
        [
            ("preprocessing", preprocessing),
            ("pca", PCA()),
            ("feature_selection", SelectKBest(score_func=f_classif)),
            ("classifier", SVC(kernel="rbf", random_state=42)),
        ]
    )

# Se configura el estimador con los hiperparámetros
def configure_estimator(pipeline):
    hyperparameters = {
        "pca__n_components": [20, 21],
        "feature_selection__k": [12],
        "classifier__kernel": ["rbf"],
        "classifier__gamma": [0.1],
    }
    
    return GridSearchCV(
        pipeline,
        hyperparameters,
        cv=10,
        n_jobs=-1,
        verbose=2,
        refit=True,
    )

# Se salva el modelo
def save_model(file_path, model):
    create_output_folder("files/models/")
    with gzip.open(file_path, "wb") as file:
        pickle.dump(model, file)

# Computar métricas
def compute_metrics(dataset_label, true_labels, predicted_labels):
    return {
        "type": "metrics",
        "dataset": dataset_label,
        "precision": precision_score(true_labels, predicted_labels, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(true_labels, predicted_labels),
        "recall": recall_score(true_labels, predicted_labels, zero_division=0),
        "f1_score": f1_score(true_labels, predicted_labels, zero_division=0),
    }

# Se computa la matriz de confusión
def compute_confusion_matrix(dataset_label, true_labels, predicted_labels):
    cm = confusion_matrix(true_labels, predicted_labels)
    return {
        "type": "cm_matrix",
        "dataset": dataset_label,
        "true_0": {"predicted_0": int(cm[0][0]), "predicted_1": int(cm[0][1])},
        "true_1": {"predicted_0": int(cm[1][0]), "predicted_1": int(cm[1][1])},
    }

# Se ejecuta el pipeline completo en pasos de preprocesamiento, entrenamiento y evaluación
def execute_pipeline():
    test_data, train_data = [preprocess_dataframe(df) for df in load_dataset("files/input")]
    
    x_train, y_train = split_features_target(train_data)
    x_test, y_test = split_features_target(test_data)
    
    pipeline = build_pipeline()
    estimator = configure_estimator(pipeline)
    estimator.fit(x_train, y_train)
    
    save_model("files/models/model.pkl.gz", estimator)
    
    create_output_folder("files/output/metrics/")
    
    y_test_pred = estimator.predict(x_test)
    test_metrics = compute_metrics("test", y_test, y_test_pred)
    train_metrics = compute_metrics("train", y_train, estimator.predict(x_train))
    
    test_confusion = compute_confusion_matrix("test", y_test, y_test_pred)
    train_confusion = compute_confusion_matrix("train", y_train, estimator.predict(x_train))
    
    with open("files/output/metrics.json", "w", encoding="utf-8") as file:
        file.write(json.dumps(train_metrics) + "\n")
        file.write(json.dumps(test_metrics) + "\n")
        file.write(json.dumps(train_confusion) + "\n")
        file.write(json.dumps(test_confusion) + "\n")

if __name__ == "__main__":
    execute_pipeline()