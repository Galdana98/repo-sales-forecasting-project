# src/utils/operators.py
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class Mapper(BaseEstimator, TransformerMixin):
    """
    Operador personalizado para mapear valores categóricos a numéricos 
    basado en un diccionario, tal como se vio en clase.
    """
    def __init__(self, variables, mappings):
        if not isinstance(variables, list):
            raise ValueError('La variable debe ser una lista')
        self.variables = variables
        self.mappings = mappings

    def fit(self, X, y=None):
        # No aprende nada de los datos, solo aplica el mapeo
        return self

    def transform(self, X):
        X = X.copy()
        for variable in self.variables:
            X[variable] = X[variable].map(self.mappings)
        return X

class DateFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Extrae características temporales (Año, Mes, Día, DíaSemana) 
    de una columna de fecha. Vital para Forecasting.
    """
    def __init__(self, variable):
        if not isinstance(variable, list):
            raise ValueError('La variable debe ser una lista')
        self.variable = variable

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for var in self.variable:
            # Asegurarse que es datetime
            X[var] = pd.to_datetime(X[var])
            
            # Crear nuevas features
            X[var + '_year'] = X[var].dt.year
            X[var + '_month'] = X[var].dt.month
            X[var + '_day'] = X[var].dt.day
            X[var + '_weekday'] = X[var].dt.dayofweek
            
            # Opcional: Eliminar la fecha original si el modelo no la acepta
            X.drop(columns=[var], inplace=True)
        return X