"""
Script de producción para Sales Forecasting.
Ejecuta un torneo de modelos (Challengers), registra todo en MLflow
y selecciona/guarda automáticamente al mejor modelo (Champion).
"""

import os
import sys
import logging
import joblib
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# Asegurar que se puede importar Utils desde el directorio superior
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from Utils.operators import DateFeatureExtractor
except ImportError:
    # Fallback por si la ejecución es local en la misma carpeta
    from operators import DateFeatureExtractor

# Configuración de Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def load_data(file_path):
    """
    Carga los datos desde un archivo Excel.
    """
    logging.info("Cargando datos desde: %s", file_path)
    try:
        df_data = pd.read_excel(file_path, sheet_name='Base de Datos')
        if 'Fecha' in df_data.columns:
            df_data['Fecha'] = pd.to_datetime(df_data['Fecha'])
            df_data.sort_values('Fecha', inplace=True)
            df_data.reset_index(drop=True, inplace=True)
        return df_data
    except Exception as error:  # pylint: disable=broad-exception-caught
        logging.error("Error al cargar datos: %s", error)
        raise

def split_data(df_data, target_col):
    """
    Divide los datos en train y validation (80/20 cronológico).
    """
    features = df_data.drop(columns=[target_col])
    target = df_data[target_col]

    split_point = int(len(df_data) * 0.80)

    x_train = features.iloc[:split_point]
    y_train = target.iloc[:split_point]
    x_val = features.iloc[split_point:]
    y_val = target.iloc[split_point:]

    return x_train, y_train, x_val, y_val

def get_pipeline_structure(model):
    """
    Crea la estructura base del pipeline con un modelo dado.
    """
    date_vars = ['Fecha']
    cat_vars = [
        'Codigo_Cupon', 'Descripcion_Cupon', 'Codigo_Producto',
        'Tipo_Orden', 'Tipo_Pago', 'Canal_Orden'
    ]
    possible_nums = ['Cantidad_Vendida', 'Precio_Menu_GTQ', 'No_Tienda']

    # Preprocesador
    preprocessor = ColumnTransformer([
        ('date', Pipeline([
            ('extractor', DateFeatureExtractor(date_vars))
        ]), date_vars),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ]), cat_vars),
        ('num', StandardScaler(), possible_nums)
    ], remainder='drop')

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])

    return pipeline

def get_models_config():
    """
    Define los modelos y sus hiperparámetros para el torneo (Challengers).
    """
    return {
        'Ridge': {
            'model': Ridge(random_state=2025),
            'params': {'model__alpha': [0.1, 1.0]}
        },
        'Lasso': {
            'model': Lasso(random_state=2025),
            'params': {'model__alpha': [0.01, 1.0]}
        },
        'RandomForest': {
            'model': RandomForestRegressor(random_state=2025, n_jobs=-1),
            # Reducido un poco para que la ejecución no sea eterna
            'params': {'model__n_estimators': [50, 100]}
        },
        'GradientBoosting': {
            'model': GradientBoostingRegressor(random_state=2025),
            'params': {'model__learning_rate': [0.1, 0.2]}
        }
    }

def run_tournament(x_train, y_train, x_val, y_val):
    """
    Ejecuta el torneo de modelos (Challengers) y devuelve el mejor (Champion).
    """
    models_config = get_models_config()
    best_rmse = float('inf')
    best_model_pipeline = None
    best_name = None

    for name, config in models_config.items():
        logging.info("Evaluando familia de modelos: %s", name)

        # Crear pipeline base con el modelo actual
        base_pipeline = get_pipeline_structure(config['model'])

        # Generar combinaciones de hiperparámetros
        param_grid = list(ParameterGrid(config['params']))

        for params in param_grid:
            run_name = f"Challenger_{name}"

            # --- MLflow Nested Run (Cada modelo es un Challenger) ---
            with mlflow.start_run(run_name=run_name, nested=True):
                # Configurar parámetros
                current_pipeline = base_pipeline.set_params(**params)

                # Entrenar
                current_pipeline.fit(x_train, y_train)

                # Evaluar
                preds = current_pipeline.predict(x_val)
                rmse = np.sqrt(mean_squared_error(y_val, preds))

                # Registrar en MLflow
                mlflow.log_params(params)
                mlflow.log_param("model_family", name)
                mlflow.log_metric("rmse", rmse)
                
                # Tag clave para identificar Challengers
                mlflow.set_tag("model_role", "Challenger")

                logging.info("  -> %s | RMSE: %.2f", params, rmse)

                # Selección del Champion (Campeón)
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_model_pipeline = current_pipeline
                    best_name = name

    return best_model_pipeline, best_rmse, best_name

def main():
    """Función principal de ejecución."""
    # Rutas
    raw_data_path = '../data/raw/data_sales_forecasting.xlsx'
    model_output_path = '../models/sales_forecasting_pipeline_prod.pkl'
    target_variable = 'Venta_Neta_GTQ'

    # Configuración MLflow
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("Sales_Forecasting_Production_Script")

    logging.info(">>> Iniciando Pipeline de Producción (Torneo de Modelos) <<<")

    try:
        # 1. Cargar y Dividir Datos
        df_data = load_data(raw_data_path)
        x_train, y_train, x_val, y_val = split_data(df_data, target_variable)

        # 2. Iniciar Corrida Padre en MLflow (Ciclo de Entrenamiento)
        with mlflow.start_run(run_name="Production_Training_Cycle"):

            # 3. Ejecutar Torneo
            champion_pipeline, champion_rmse, champion_name = run_tournament(
                x_train, y_train, x_val, y_val
            )

            logging.info(">>> CAMPEÓN DEFINITIVO: %s con RMSE: %.2f <<<",
                         champion_name, champion_rmse)

            # 4. Registrar al Champion en MLflow (Inciso b)
            mlflow.log_metric("champion_rmse", champion_rmse)
            mlflow.set_tag("winner_model", champion_name)
            mlflow.set_tag("model_role", "Champion")

            # Guardar el artefacto en MLflow
            mlflow.sklearn.log_model(champion_pipeline, "champion_model_artifact")

            # 5. Guardar modelo físico localmente
            os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
            joblib.dump(champion_pipeline, model_output_path)
            logging.info("Pipeline guardado exitosamente en: %s", model_output_path)

        logging.info(">>> Pipeline finalizado exitosamente <<<")

    except Exception as main_error:  # pylint: disable=broad-exception-caught
        logging.critical("El proceso falló: %s", main_error)
        sys.exit(1)

if __name__ == "__main__":
    main()