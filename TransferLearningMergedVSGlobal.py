import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
import keras 
import os
import json

import random as python_random
import time
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.metrics import MeanAbsolutePercentageError
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.models import load_model

from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor

from keras import layers, models

np.random.seed(123)
python_random.seed(123)
tf.random.set_seed(1234)

from keras_tuner import HyperModel, HyperParameters
from keras_tuner.tuners import Hyperband

from funcionesComunes import *


class LSTMHyperModel(HyperModel):
    
    def __init__(self, input_shape, num_lstm_layers_input=3, num_lstm_layers_after_input=3):
        self.input_shape = input_shape
        self.num_lstm_layers_input = num_lstm_layers_input
        self.num_lstm_layers_after_input = num_lstm_layers_after_input
    
    def build(self, hp):
        
        # Podemos ver como "hp.Int" permite ir modificando los parámetros del modelo con INT
        # Con "hp.Float" lo hace pero para valores decimales 
        # IMPORTANTE!!! Nunca se devuelve el return_sequences si la LSTM es la última capa
        # IMPORTANTE!!! Si estamos aplicando capas LSTM despues Dropout y luego LSTM de nuevo, debemos de poner en todas las primeras capas LSTM return_sequences=True !!!IMPORTANTE
        
        model = Sequential()
        
        # Número de capas LSTM
        num_lstm_layers = hp.Int('num_lstm_layers', min_value=1, max_value=self.num_lstm_layers_input, default = self.num_lstm_layers_input)

        # Seleccionamos el batch_size usado
        self.batch_size=hp.Choice("batch_size", [16, 24, 32])

        for i in range(num_lstm_layers):
            # Añadiendo capas LSTM
            # Solo la primera capa necesita input_shape
            if i == 0:
                
                # Se tiene una capa LSTM que empieza con 32 unidades y va aumentando de 32 en 32 hasta 128
                model.add(LSTM(units=hp.Int(f'units_lstm_{i}', min_value=32, max_value=128, step=32),
                               input_shape=self.input_shape,
                               return_sequences=True,  # True si hay más de una capa LSTM
                               use_bias=True))
            else:
                model.add(LSTM(units=hp.Int(f'units_lstm_{i}', min_value=32, max_value=128, step=32),
                               return_sequences=True,  # True para todas excepto la última capa LSTM
                               use_bias=True))
        
        # Se modifica la capa Dropout desde 0.0 hasta 0.5 con un step de 0.05
        model.add(Dropout(hp.Float('dropout', min_value=0.0, max_value=0.5, default=0.25, step=0.05)))
        
        # Creamos un bucle que permite incrementar las capas LSTM destras de la capa Dropout
        num_lstm_layers_after = hp.Int('num_lstm_layers_after', min_value=1, max_value=self.num_lstm_layers_after_input, default = self.num_lstm_layers_after_input)
        
        for i in range(num_lstm_layers_after):
            # Añadiendo capas LSTM después de Dropout
            model.add(LSTM(units=hp.Int(f'units_lstm_after_{i}', min_value=32, max_value=128, step=32),
                           return_sequences=(i < num_lstm_layers_after - 1),  # True para todas excepto la última capa LSTM
                           use_bias=True))
        
        # Capa dense que empieza con 4 unidades y va aumentando de 4 en 4 hasta 64
        model.add(Dense(hp.Int('dense_units', min_value=4, max_value=64, step=4), activation=hp.Choice('dense_activation',values=['relu', 'sigmoid', 'tanh', 'elu', 'relu'],default='relu'), use_bias=True))
        
        # Se modifica la capa Dropout desde 0.0 hasta 0.5 con un step de 0.05
        model.add(Dropout(hp.Float('dropout_2', min_value=0.0, max_value=0.5, default=0.25, step=0.05)))

        model.add(Dense(1, activation=hp.Choice('dense_activation',values=['relu', 'sigmoid', 'tanh', 'elu', 'relu'],default='relu'), use_bias=True))
        
        # Variaciones de los optimizadores
        optimizer = hp.Choice('optimizer', ['adam', 'sgd', 'rmsprop'])
        if optimizer == 'adam':
            opt = Adam(
                learning_rate=hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='LOG')
            )
        elif optimizer == 'sgd':
            opt = SGD(
                learning_rate=hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='LOG')
            )
        else: # rmsprop
            opt = RMSprop(
                learning_rate=hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='LOG')
            )
        
        
        model.compile(optimizer=opt,
                      loss=MeanSquaredError(),
                      metrics=[
                        MeanSquaredError(name='mse'),
                        RootMeanSquaredError(name='rmse'),
                        MeanAbsolutePercentageError(name='mape')
                        ]
                    )
        
        return model
    
    def fit(self, hp, model, *args, **kwargs):
        return model.fit(*args,
            batch_size = self.batch_size,
            **kwargs)

# Definimos la función que se ejecutará en paralelo
def adjustHyperparameters_Paralelized(archivo, WINDOWS_SIZE=3):
    
    if "train" in archivo:

        nombreArchivo = archivo.split("_train_transfer")[0]
        print(f"Estamos procesando el archivo {archivo}")

        data_train_global = np.load(f'RCPMergedTransferTotal/{nombreArchivo}_train_transfer.npz', allow_pickle=True)
        data_val_global = np.load(f'RCPMergedTransferTotal/{nombreArchivo}_val_transfer.npz', allow_pickle=True)
        data_test_global = np.load(f'RCPMergedTransferTotal/{nombreArchivo}_val.npz', allow_pickle=True)

        X_train = data_train_global['X_train_transfer']
        y_train = data_train_global['y_train_transfer']
        X_val = data_val_global['X_val_transfer']
        y_val = data_val_global['y_val_transfer']
        X_test = data_test_global['X_val']
        y_test = data_test_global['y_val']

        # Los valores de normalización estan incluidos en todos los archivos (train, val, test) y cada uno de ellos tiene todo el contenido del resto
        valorNormalizacion = data_train_global["valorNormalizacion"]

        hypermodel = LSTMHyperModel(input_shape=(WINDOWS_SIZE + 1, X_train.shape[2]), num_lstm_layers_input=3, num_lstm_layers_after_input=3)

        tuner = Hyperband(
            hypermodel,
            objective='val_loss',
            max_epochs=50,
            factor=3,
            directory='best_models_22_10_24/LSTM',
            project_name=f'hyperparameter_tuning_{nombreArchivo}'
        )

        tuner.search(X_train, y_train, epochs=50, validation_data=(X_val, y_val))

        top_models_data = []

        best_trials = tuner.oracle.get_best_trials(num_trials=10)

        for i, trial in enumerate(best_trials):
            hyperparameters = trial.hyperparameters.values
            model = tuner.hypermodel.build(trial.hyperparameters)
            model.load_weights(tuner.get_trial_dir(trial.trial_id) + "/checkpoint.weights.h5")
            
            # Guardamos el modelo
            model.save(f"models/LSTMMerged_22_10_24/{os.path.splitext(nombreArchivo)[0]}_model_{i+1}.keras")

            # Entrenamiento
            predictions_train = predictionForIndividuals(X_train, y_train, model, hyperparameters["batch_size"])
            predictions_train["PredictionsDenormalize"] = predictions_train.apply(lambda row: desnormalizacionBAI(row, valorNormalizacion, "Predictions"), axis=1)
            predictions_train["ActualDenormalize"] = predictions_train.apply(lambda row: desnormalizacionBAI(row, valorNormalizacion, "Actuals"), axis=1)

            train_mse = mean_squared_error(predictions_train["ActualDenormalize"], predictions_train["PredictionsDenormalize"])
            train_rmse = np.sqrt(train_mse)
            train_mape = mean_absolute_percentage_error(predictions_train["ActualDenormalize"], predictions_train["PredictionsDenormalize"]) *100
            train_r2 = r2_score(predictions_train["ActualDenormalize"], predictions_train["PredictionsDenormalize"])

            # Validación
            predictions_val = predictionForIndividuals(X_val, y_val, model, hyperparameters["batch_size"])
            predictions_val["PredictionsDenormalize"] = predictions_val.apply(lambda row: desnormalizacionBAI(row, valorNormalizacion, "Predictions"), axis=1)
            predictions_val["ActualDenormalize"] = predictions_val.apply(lambda row: desnormalizacionBAI(row, valorNormalizacion, "Actuals"), axis=1)

            val_mse = mean_squared_error(predictions_val["ActualDenormalize"], predictions_val["PredictionsDenormalize"])
            val_rmse = np.sqrt(val_mse)
            val_mape = mean_absolute_percentage_error(predictions_val["ActualDenormalize"], predictions_val["PredictionsDenormalize"]) *100
            val_r2 = r2_score(predictions_val["ActualDenormalize"], predictions_val["PredictionsDenormalize"])

            # Prueba
            predictions_test = predictionForIndividuals(X_test, y_test, model, hyperparameters["batch_size"])
            predictions_test["PredictionsDenormalize"] = predictions_test.apply(lambda row: desnormalizacionBAI(row, valorNormalizacion, "Predictions"), axis=1)
            predictions_test["ActualDenormalize"] = predictions_test.apply(lambda row: desnormalizacionBAI(row, valorNormalizacion, "Actuals"), axis=1)

            test_mse = mean_squared_error(predictions_test["ActualDenormalize"], predictions_test["PredictionsDenormalize"])
            test_rmse = np.sqrt(test_mse)
            test_mape = mean_absolute_percentage_error(predictions_test["ActualDenormalize"], predictions_test["PredictionsDenormalize"]) *100
            test_r2 = r2_score(predictions_test["ActualDenormalize"], predictions_test["PredictionsDenormalize"])

            # Almacenar la información del modelo
            model_info = {
                'Model Rank': i+1,
                'Trial ID': trial.trial_id,
                # Métricas del conjunto de entrenamiento
                'train_mse': train_mse,
                'train_rmse': train_rmse,
                'train_mape': train_mape,
                'train_r2': train_r2,
                # Métricas del conjunto de validación
                'val_mse': val_mse,
                'val_rmse': val_rmse,
                'val_mape': val_mape,
                'val_r2': val_r2,
                # Métricas del conjunto de prueba
                'test_mse': test_mse,
                'test_rmse': test_rmse,
                'test_mape': test_mape,
                'test_r2': test_r2
            }

            print(f"RESULTADOS DE MSE, RMSE, R2, MAPE (Train): {train_mse}, {train_rmse}, {train_r2}, {train_mape}")
            print(f"RESULTADOS DE MSE, RMSE, R2, MAPE (Val): {val_mse}, {val_rmse}, {val_r2}, {val_mape}")
            print(f"RESULTADOS DE MSE, RMSE, R2, MAPE (Test): {test_mse}, {test_rmse}, {test_r2}, {test_mape}")

            # Agregamos los hyperparametros
            model_info.update(hyperparameters)
            top_models_data.append(model_info)

        with open(f"models/LSTMMerged_22_10_24/{nombreArchivo}_best_models.json", 'w') as file:
            json.dump(top_models_data, file, indent=4)



if __name__ == "__main__":

    # Importamos la libreria de multiparalelización
    import multiprocessing as mp

    archivos = [archivo for archivo in os.listdir("RCPMergedTransferTotal")]

    # Crear un grupo de procesos
    print(f"VAMOS A EJECUTAR EL PROCESO CON {mp.cpu_count()} NUCLEOS")
    with mp.Pool(processes=mp.cpu_count()) as pool:
        pool.map(adjustHyperparameters_Paralelized, archivos)
