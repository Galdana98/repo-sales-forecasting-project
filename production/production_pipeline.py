"""
Script de ejecución del pipeline de entrenamiento para Sales Forecasting.
Integra MLflow para tracking automático de experimentos.
"""

import os
import sys
import logging
import joblib
import pandas as pd
import numpy as np
import mlflow # UPGRADE: Importar MLflow
import mlflow.sklearn # UPGRADE: Importar módulo de sklearn para autolog
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Asegurar que se puede importar Utils desde el directorio superior
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Importar operadores personalizados
try:
    from Utils.operators import DateFeatureExtractor
except ImportError:
    from operators import DateFeatureExtractor

# Configuración de Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def load_data(file_path):
    """Carga los datos desde un archivo Excel."""
    logging.info("Cargando datos desde: %s", file_path)
    try:
        df_data = pd.read_excel(file_path, sheet_name='Base de Datos')
        if 'Fecha' in df_data.columns:
            df_data['Fecha'] = pd.to_datetime(df_data['Fecha'])
            df_data.sort_values('Fecha', inplace=True)
            df_data.reset_index(drop=True, inplace=True)
        logging.info("Datos cargados exitosamente. Dimensiones: %s", df_data.shape)
        return df_data
    except FileNotFoundError:
        logging.error("Error: No se encontró el archivo %s", file_path)
        raise
    except Exception as error: # pylint: disable=broad-exception-caught
        logging.error("Error inesperado al cargar datos: %s", error)
        raise

def split_data(df_data, target_col):
    """Divide los datos en train y validation."""
    logging.info("Dividiendo datos en Train y Validation...")
    features = df_data.drop(columns=[target_col])
    target = df_data[target_col]

    split_point = int(len(df_data) * 0.80)

    x_train = features.iloc[:split_point]
    y_train = target.iloc[:split_point]
    x_val = features.iloc[split_point:]
    y_val = target.iloc[split_point:]

    logging.info("Split completado. Train: %s, Val: %s", x_train.shape, x_val.shape)
    return x_train, y_train, x_val, y_val

def build_pipeline(x_train):
    """Construye el pipeline de preprocesamiento y modelado."""
    logging.info("Configurando el Pipeline...")
    date_vars = ['Fecha']
    cat_vars = [
        'Codigo_Cupon', 'Descripcion_Cupon', 'Codigo_Producto',
        'Tipo_Orden', 'Tipo_Pago', 'Canal_Orden'
    ]
    possible_nums = ['Cantidad_Vendida', 'Precio_Menu_GTQ', 'No_Tienda']
    num_vars = [c for c in possible_nums if c in x_train.columns]

    preprocessor = ColumnTransformer([
        ('date', Pipeline([
            ('extractor', DateFeatureExtractor(date_vars))
        ]), date_vars),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ]), cat_vars),
        ('num', StandardScaler(), num_vars)
    ], remainder='drop')

    model_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', RandomForestRegressor(n_estimators=150, random_state=2025, n_jobs=-1))
    ])

    return model_pipeline

def train_and_evaluate(pipeline, x_train, y_train, x_val, y_val):
    """Entrena el pipeline y evalúa su desempeño."""
    logging.info("Iniciando entrenamiento del modelo...")
    pipeline.fit(x_train, y_train)
    logging.info("Entrenamiento finalizado.")

    logging.info("Evaluando modelo...")
    preds = pipeline.predict(x_val)
    rmse = np.sqrt(mean_squared_error(y_val, preds))
    logging.info("Validación RMSE: %.2f", rmse)
    
    # UPGRADE: Registrar métrica manual adicional si se desea (opcional con autolog)
    mlflow.log_metric("validation_rmse", rmse) #

def save_pipeline(pipeline, output_path):
    """Guarda el pipeline entrenado en disco."""
    logging.info("Guardando artefacto localmente en: %s", output_path)
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        joblib.dump(pipeline, output_path)
        logging.info("Pipeline guardado exitosamente.")
    except OSError as error:
        logging.error("Error de sistema al guardar el pipeline: %s", error)
        raise

def main():
    """Función principal de ejecución."""
    # Configuración de rutas
    raw_data_path = '../data/raw/data_sales_forecasting.xlsx'
    model_output_path = '../models/sales_forecasting_pipeline_prod.pkl'
    target_variable = 'Venta_Neta_GTQ'

    logging.info(">>> Iniciando Pipeline de Producción MLOps <<<")

    # UPGRADE: Configuración de MLflow
    mlflow.set_tracking_uri("http://127.0.0.1:5000") 
    mlflow.set_experiment("Sales_Forecasting_Experiment")
    
    # UPGRADE: Activar Tracking Automático (Autolog)
    # Esto registra params, métricas y el modelo automáticamente
    mlflow.sklearn.autolog()

    try:
        # UPGRADE: Iniciar la corrida de MLflow
        with mlflow.start_run(run_name="RandomForest_Prod_Run"):
            
            # 1. Cargar Datos
            df_data = load_data(raw_data_path)

            # 2. Split de Datos
            x_train, y_train, x_val, y_val = split_data(df_data, target_variable)

            # 3. Construir Pipeline
            pipeline = build_pipeline(x_train)

            # 4. Entrenar y Evaluar (Todo lo que ocurra aquí se registrará en MLflow)
            train_and_evaluate(pipeline, x_train, y_train, x_val, y_val)

            # 5. Guardar Modelo Local
            save_pipeline(pipeline, model_output_path)
            
            # (El modelo también se guarda automáticamente en MLflow gracias a autolog)

        logging.info(">>> Pipeline ejecutado y registrado en MLflow correctamente <<<")

    except Exception as main_error: # pylint: disable=broad-exception-caught
        logging.critical("El proceso falló: %s", main_error)
        sys.exit(1)

if __name__ == "__main__":
    main()