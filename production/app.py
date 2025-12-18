"""
API Flask para Sales Forecasting.
Permite predicciones individuales y por lote (batch).
"""

import os
import sys
import datetime
import logging
import joblib
import pandas as pd
from flask import Flask, request, jsonify

# --- Configuración de Rutas y Logging ---
# Asegurar que se puede importar Utils desde el directorio superior
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from Utils.operators import DateFeatureExtractor
except ImportError:
    from operators import DateFeatureExtractor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)

# --- Carga del Modelo y Configuración ---
MODEL_PATH = '../models/sales_forecasting_pipeline_prod.pkl'
METRICS = {"validation_rmse": 5.46}

try:
    pipeline = joblib.load(MODEL_PATH)
    logging.info("Modelo cargado exitosamente desde: %s", MODEL_PATH)
except Exception as e:
    logging.error("Error al cargar el modelo: %s", e)
    pipeline = None


def get_model_params(pipeline_obj):
    """Extrae los hiperparámetros principales del modelo final del pipeline."""
    try:
        model_step = pipeline_obj.named_steps['model']
        return model_step.get_params()
    except Exception:
        return "No disponibles"

# --- Endpoints ---


@app.route('/')
def home():
    """Ruta base para verificar que la API está viva."""
    return "<h1>Sales Forecasting API is Running!</h1>"


@app.route('/predict_individual', methods=['POST'])
def predict_individual():
    """
    a. Endpoint para predicción de un registro individual.
    Espera un JSON con los campos de una sola fila.
    """
    try:
        # 1. Obtener datos del request
        data = request.get_json()

        # 2. Convertir a DataFrame (1 fila)
        df_input = pd.DataFrame([data])

        if 'Fecha' in df_input.columns:
            df_input['Fecha'] = pd.to_datetime(df_input['Fecha'])

        # 3. Predecir
        prediction = pipeline.predict(df_input)[0]

        # 4. Construir respuesta
        response = {
            'timestamp': datetime.datetime.now().isoformat(),
            'prediction': float(prediction),
            'model_metrics': METRICS,
            'hyperparameters': get_model_params(pipeline)
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    """
    b. Endpoint para predicción múltiple (batch).
    Espera una lista de objetos JSON.
    """
    try:
        # 1. Obtener datos (lista de diccionarios)
        data = request.get_json()

        # 2. Convertir a DataFrame
        df_input = pd.DataFrame(data)

        if 'Fecha' in df_input.columns:
            df_input['Fecha'] = pd.to_datetime(df_input['Fecha'])

        # 3. Predecir
        predictions = pipeline.predict(df_input)

        # 4. Construir respuesta
        response = {
            'timestamp': datetime.datetime.now().isoformat(),
            'batch_size': len(predictions),
            'predictions': predictions.tolist(),  # Convertir numpy array a lista
            'model_metrics': METRICS,
            'hyperparameters': get_model_params(pipeline)
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    app.run(debug=True, port=5000)
