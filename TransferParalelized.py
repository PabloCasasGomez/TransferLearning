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

from funcionesComunes import *

import multiprocessing

################################# INICIO DEL PROCESO ################################################

# Creamos un df que contiene los resultados de los modelos freeze de cada especie
df_Freeze = pd.DataFrame(columns = ["File", "Time", "TrainMSE", "TrainRMSE", "TrainR2", "TrainMAPE", 
                                    "ValidationMSE", "ValidationRMSE", "ValidationR2", "ValidationMAPE",
                                    "TestMSE", "TestRMSE", "TestR2", "TestMAPE"])

# Leemos los diferentes modelos
for model in os.listdir("models/LSTM"):

    if model != "model_Abies_spectabilis.keras":
        print(f'Procesando el modelo: {model}')

        # Para cada archivo 
        for archivo in os.listdir("RCPMerged"):

            # Comprobamos que el modelo no sea del archivo a procesar
            archivoProcesar = archivo.split("_merged")[0]
            modeloProcesar = model.split("model_")[1].split(".keras")[0]
            print(archivoProcesar, modeloProcesar)

            # Lanzamos la ejecución siempre que no sea el mismo archivo o el global
            if archivoProcesar != modeloProcesar and archivo != "totalMerged.csv":

                # Cargar el archivo CSV con punto y coma como delimitador
                df = pd.read_csv(f'RCPMerged/{archivo}')
                df = df[~df["nametag"].str.startswith("INDI005")]

                # Codificamos, normalización y split de datos
                df = codification(df)
                print(df.shape)

                df, valorNormalizacion = individualNormalization(df)
                print(f"SE HA NORMALIZADO EL ARCHIVO: {archivo}")

                temp_df = df
                train_data, val_data, test_data = split_population_individuals(temp_df, train_pct=0.80, val_pct_in_train=0.20, details=False)
                train_data.shape, val_data.shape, test_data.shape

                # Obtenemos X e y para los datasets de train, val y test 
                WINDOWS_SIZE = 3
                X_train, y_train = df_to_X_y_ind_3(train_data, WINDOWS_SIZE)
                X_val, y_val = df_to_X_y_ind_3(val_data, WINDOWS_SIZE)
                X_test, y_test = df_to_X_y_ind_3(test_data, WINDOWS_SIZE)
                print(X_train.shape, X_test.shape, X_val.shape)

                # Cargamos el modelo global
                modelLSTM = tf.keras.models.load_model(f'models/LSTM/{model}')
                modelLSTMFreeze = modelLSTM

                # Obtener el optimizador del modelo
                optimizer = modelLSTMFreeze.optimizer

                # Obtenemos el valor de batch size
                with open(f'arquitecturaModelosJSON/LSTM/{archivo.split(".csv")[0]}_best_models_updated_no_train.json') as f:
                    parameters = json.load(f)

                batch_size_LSTM = parameters[0]["batch_size"]

                # Indicamos el numero de layers a entrenar
                NUM_TRAINABLE = 1

                numLSTM_layers = sum(1 for layer in modelLSTMFreeze.layers if "lstm" in layer.name)

                numFreezeLayers = numLSTM_layers - NUM_TRAINABLE # Congelamos todas menos la útlima capa

                numberLayersFreezed = 0

                # Congelar las primeras 'numFreezeLayers' capas LSTM
                for i, layer in enumerate(modelLSTMFreeze.layers):

                    if "lstm" in layer.name and numFreezeLayers>numberLayersFreezed:
                        numberLayersFreezed += 1
                        layer.trainable = False

                    print(layer, layer.trainable)

                # Compilamos el modelo con los nuevos datos
                modelLSTMFreeze.compile(optimizer=optimizer, loss=MeanSquaredError(), metrics=[
                                            MeanSquaredError(name='mse'),
                                            RootMeanSquaredError(name='rmse'),
                                            MeanAbsolutePercentageError(name='mape')
                                        ])

                # Comienza a medir el tiempo de entrenamiento
                start_time = time.time()

                historyLSTMTransfer = modelLSTMFreeze.fit(X_train, y_train, epochs=200, batch_size=batch_size_LSTM,
                                        validation_data=(X_val, y_val), callbacks=[EarlyStopping(monitor='val_loss', patience=10)], verbose=0)

                # Finaliza la medición del tiempo de entrenamiento
                end_time = time.time()
                print(f"[MODELO CONGELADO]: {end_time-start_time}")

                # Realizar predicciones y calcular métricas para el conjunto de entrenamiento
                predictions_train = predictionForIndividuals(X_train, y_train, modelLSTMFreeze, batch_size_LSTM)
                predictions_train["PredictionsDenormalize"] = predictions_train.apply(lambda row: desnormalizacionBAI(row, valorNormalizacion, "Predictions"), axis=1)
                predictions_train["ActualDenormalize"] = predictions_train.apply(lambda row: desnormalizacionBAI(row, valorNormalizacion, "Actuals"), axis=1)

                train_mse = mean_squared_error(predictions_train["ActualDenormalize"],predictions_train["PredictionsDenormalize"])
                train_rmse = np.sqrt(train_mse)
                train_mape = (np.sum(np.abs(predictions_train["PredictionsDenormalize"] - predictions_train["ActualDenormalize"])) / np.sum(np.abs(predictions_train["ActualDenormalize"]))) * 100
                train_r2 = r2_score(predictions_train["ActualDenormalize"], predictions_train["PredictionsDenormalize"])

                # Realizar predicciones y calcular métricas para el conjunto de validación
                predictions_val = predictionForIndividuals(X_val, y_val, modelLSTMFreeze, batch_size_LSTM)
                predictions_val["PredictionsDenormalize"] = predictions_val.apply(lambda row: desnormalizacionBAI(row, valorNormalizacion, "Predictions"), axis=1)
                predictions_val["ActualDenormalize"] = predictions_val.apply(lambda row: desnormalizacionBAI(row, valorNormalizacion, "Actuals"), axis=1)

                val_mse = mean_squared_error(predictions_val["ActualDenormalize"],predictions_val["PredictionsDenormalize"])
                val_rmse = np.sqrt(val_mse)
                val_mape = (np.sum(np.abs(predictions_val["PredictionsDenormalize"] - predictions_val["ActualDenormalize"])) / np.sum(np.abs(predictions_val["ActualDenormalize"]))) * 100
                val_r2 = r2_score(predictions_val["ActualDenormalize"], predictions_val["PredictionsDenormalize"])

                # Realizar predicciones y calcular métricas para el conjunto de prueba
                predictions_test = predictionForIndividuals(X_test, y_test, modelLSTMFreeze, batch_size_LSTM)
                predictions_test["PredictionsDenormalize"] = predictions_test.apply(lambda row: desnormalizacionBAI(row, valorNormalizacion, "Predictions"), axis=1)
                predictions_test["ActualDenormalize"] = predictions_test.apply(lambda row: desnormalizacionBAI(row, valorNormalizacion, "Actuals"), axis=1)

                test_mse = mean_squared_error(predictions_test["ActualDenormalize"],predictions_test["PredictionsDenormalize"])
                test_rmse = np.sqrt(test_mse)
                test_mape = (np.sum(np.abs(predictions_test["PredictionsDenormalize"] - predictions_test["ActualDenormalize"])) / np.sum(np.abs(predictions_test["ActualDenormalize"]))) * 100
                test_r2 = r2_score(predictions_test["ActualDenormalize"], predictions_test["PredictionsDenormalize"])

                print(f"RESULTADOS DE MSE, RMSE, R2, MAPE (Train): {train_mse}, {train_rmse}, {train_r2}, {train_mape}")
                print(f"RESULTADOS DE MSE, RMSE, R2, MAPE (Val): {val_mse}, {val_rmse}, {val_r2}, {val_mape}")
                print(f"RESULTADOS DE MSE, RMSE, R2, MAPE (Test): {test_mse}, {test_rmse}, {test_r2}, {test_mape}")

                # Guardamos los datos calculados
                df_Freeze.loc[len(df_Freeze)] = [archivo, end_time-start_time,train_mse, train_rmse, train_r2, train_mape, val_mse, val_rmse, val_r2, val_mape, test_mse, test_rmse, test_r2, test_mape]

        df_Freeze.to_csv(f'resultadosTransfer_{archivo.split("_merged")[0]}.csv', index=False)


# Preparar lista de tareas para paralelización
tareas = [(model, archivo) for model in os.listdir("models/LSTM") if model != "model_Abies_spectabilis.keras"
          for archivo in os.listdir("RCPMerged")]

# Ejecutar el procesamiento en paralelo utilizando multiprocessing
if __name__ == '__main__':

    num_workers = min(multiprocessing.cpu_count(), len(tareas))  # Limitar el número de procesos al menor de núcleos y tareas


    # Creamos procesos
    with multiprocessing.Pool() as pool:
