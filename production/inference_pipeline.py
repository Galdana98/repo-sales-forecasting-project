"""
Script de Inferencia para DVC.
Carga el modelo entrenado y genera predicciones sobre datos nuevos (batch).
"""

import os
import sys
import logging
from datetime import datetime
import joblib
import pandas as pd

# --- Configuración de Rutas para importar Utils ---
# Esto agrega la carpeta raíz del proyecto al sistema para poder ver 'Utils'
# pylint: disable=wrong-import-position
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    # Importamos la clase necesaria para que joblib reconozca el objeto dentro del pickle
    # pylint: disable=import-error
    from Utils.operators import DateFeatureExtractor
except ImportError:
    logging.error("No se pudo importar Utils.operators. Verifica la estructura del proyecto.")
    sys.exit(1)

# Configuración de Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Rutas (Absolutas para DVC desde la raíz)
MODEL_PATH = 'models/sales_forecasting_pipeline_prod.pkl'
DATA_PATH = 'data/raw/data_sales_forecasting.xlsx'
OUTPUT_PATH = 'data/predictions/batch_predictions.csv'

def run_inference():
    """
    Ejecuta el flujo de inferencia: carga modelo, lee datos, predice y guarda.
    """
    logging.info(">>> Iniciando Proceso de Inferencia Batch <<<")

    # 1. Verificar si existe el modelo
    if not os.path.exists(MODEL_PATH):
        logging.error("No se encontró modelo en: %s. Ejecuta 'dvc repro'.", MODEL_PATH)
        sys.exit(1)

    try:
        # 2. Cargar el Modelo
        # Al cargar, joblib necesita ver la definición de DateFeatureExtractor
        pipeline = joblib.load(MODEL_PATH)
        logging.info("Modelo cargado exitosamente.")

        # 3. Cargar Datos
        logging.info("Cargando datos para inferencia...")
        if not os.path.exists(DATA_PATH):
            logging.error("No se encontraron datos en: %s", DATA_PATH)
            sys.exit(1)

        df_new = pd.read_excel(DATA_PATH, sheet_name='Base de Datos')

        # Simulamos predicción sobre los últimos 50 registros
        df_batch = df_new.tail(50).copy()

        # Asegurar formato de fecha
        if 'Fecha' in df_batch.columns:
            df_batch['Fecha'] = pd.to_datetime(df_batch['Fecha'])

        # 4. Generar Predicciones
        logging.info("Generando predicciones...")
        predictions = pipeline.predict(df_batch)

        # 5. Guardar Resultados
        df_batch['Prediccion_Ventas'] = predictions
        df_batch['Fecha_Ejecucion'] = datetime.now()

        # Crear carpeta si no existe
        os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

        df_batch.to_csv(OUTPUT_PATH, index=False)
        logging.info("Predicciones guardadas en: %s", OUTPUT_PATH)
        logging.info(">>> Inferencia finalizada exitosamente <<<")

    except Exception as e:  # pylint: disable=broad-exception-caught
        logging.critical("Error durante la inferencia: %s", e)
        sys.exit(1)

if __name__ == "__main__":
    run_inference()