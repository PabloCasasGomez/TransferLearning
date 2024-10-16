import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras import layers, models
import tensorflow as tf
import math
import os
import json


def filtradoIndividuos(X, individual):

    # Crea una máscara booleana para todas las filas donde la segunda columna de la segunda dimensión es igual al individuo seleccionado
    mask = X[:, :, 1] == individual

    # Dado que la máscara es de dos dimensiones, expande la máscara para que tenga tres dimensiones
    # (esto es necesario para mantener la estructura tridimensional durante el filtrado)
    mask_expanded = np.expand_dims(mask, axis=2)

    # Repite la máscara para todas las columnas de la tercera dimensión
    mask_repeated = np.repeat(mask_expanded, X.shape[2], axis=2)

    # Aplica la máscara extendida a X para obtener X_indv
    X_indv = X[mask_repeated].reshape(-1, X.shape[1], X.shape[2])
    
    return X_indv

def predictionForIndividuals(X, y, model, batch_size):
    # Quiero crear una predicción por cada individuo
    uniqueIndividuals = np.unique(X[:,:,1])

    # Creamos el dataset para todos los datos predichos
    predictionsDF = pd.DataFrame(columns=["year", "nametag_encoded", "Predictions", "Actuals"])

    for individual in uniqueIndividuals:
        
        # Filtrar los datos del individuo actual
        X_indv = filtradoIndividuos(X, individual)
        
        # Creamos la predicción de los datos para el individuo actual
        y_pred = model.predict(X_indv, batch_size=batch_size, verbose = 0)

        # Creamos un DataFrame temporal con los datos
        tempDF = pd.DataFrame({
            "year": X_indv[:,0,2].flatten(), 
            "nametag_encoded": X_indv[:,0,1].flatten(), 
            "Predictions": y_pred.flatten(),  # Asegúrate de que y_pred tenga la misma longitud que los otros
            "Actuals": y[X[:,0,1] == individual].flatten()
        })
        
        # Concatenamos el DataFrame temporal con el principal
        predictionsDF = pd.concat([predictionsDF, tempDF], ignore_index=True)
        
    return predictionsDF

# Función para desnormalizar BAI
def desnormalizacionBAI(row, valorNormalizacion, columnaObjetivo):
    if row['nametag_encoded'] in valorNormalizacion:
        return row[columnaObjetivo] * valorNormalizacion[row['nametag_encoded']]
    else:
        return row[columnaObjetivo]