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

def codification(df):
    # Creamos columna de poblaciones a partir de nametag y solo quedandonos con los 7 primeros caracteres
    df["poblacion"]=df["nametag"].str[:7]

    # Codificacion de los nametags y nombre de especie
    label_encoder = LabelEncoder()

    df['poblacion_encoded'] = label_encoder.fit_transform(df['poblacion'])
    df['nametag_encoded'] = label_encoder.fit_transform(df['nametag'])

    # Eliminamos las columnas originales
    df.drop(['poblacion','nametag'], axis=1, inplace=True)

    # Reordenamos las columnas
    otrasColumnas=[columna for columna in df.columns if columna not in ["poblacion_encoded","nametag_encoded"]]

    # Reordenamos el dataset
    df=df[["poblacion_encoded","nametag_encoded"]+otrasColumnas]
    
    return df

def individualNormalization(df):

    valorNormalización = {}

    for individuo in df["nametag_encoded"].unique():

        # Obtenemos el valor máximo de bai para la serie en cuestion
        baiMax = df[df["nametag_encoded"]==individuo]["bai"].max()

        # Incluimos el valor de bai para desnormalizar despues
        valorNormalización[individuo] = baiMax

        # Dividimos los valores de bai del individuo por el máximo de la serie del árbol
        df.loc[df["nametag_encoded"]==individuo, "bai"] /= baiMax
    
    return df, valorNormalización

def split_population_individuals(X, train_pct, val_pct_in_train, details = True, random_state=42):
    train_data, val_data, test_data = [], [], []

    # Identificar las poblaciones únicas
    populations = X["poblacion_encoded"].unique()
    if details:
        print("Las poblaciones son: ", populations)

    for pop in populations:
        if details:
            print("Se esta procesando la poblacion: ", pop)
        # Extraer los datos de la población actual
        pop_data = X[X["poblacion_encoded"] == pop]

        # Identificar los individuos únicos en esta población
        individuals = pop_data["nametag_encoded"].unique()
        
        if details:
            print("Los individuos de esta poblacion son: ", individuals)

        # Dividir los individuos en train+val y test
        individuals_train_val, individuals_test = train_test_split(individuals, train_size=train_pct, random_state=random_state)

        # Dividir los individuos de train+val en train y val
        individuals_train, individuals_val = train_test_split(individuals_train_val, test_size=val_pct_in_train, random_state=random_state)

        # Agregar los datos correspondientes a los individuos seleccionados
        for ind in individuals_train:
            ind_data = pop_data[pop_data["nametag_encoded"] == ind]
            train_data.append(ind_data)

        for ind in individuals_val:
            ind_data = pop_data[pop_data["nametag_encoded"] == ind]
            val_data.append(ind_data)

        for ind in individuals_test:
            ind_data = pop_data[pop_data["nametag_encoded"] == ind]
            test_data.append(ind_data)

    # Convertir listas a arrays
    train_data = pd.concat(train_data)
    val_data = pd.concat(val_data)
    test_data = pd.concat(test_data)
    
    return train_data, val_data, test_data

def df_to_X_y_ind_3(df, window_size=3):
    X = []
    y = []

    # Identificamos la posición de la columna a predecir "bai"
    bai_index = df.columns.get_loc("bai")
    
    # Identificamos las columnas con "model" en su nombre
    model_columns = [col for col in df.columns if "model" in col]
    model_indices = [df.columns.get_loc(col) for col in model_columns]

    # Identificamos las columnas sin "model" en su nombre
    non_model_indices = [k for k in range(df.shape[1]) if k not in model_indices]

    # Columnas con descriptores 
    descriptor_keywords = ["poblacion_encoded", "nametag_encoded", "Year", "elevation", "latitude", "longitude", "age"]
    descriptor_indices = [df.columns.get_loc(col) for col in descriptor_keywords if col in df.columns]

    # Identificamos los individuos únicos en el dataframe
    individuals = df["nametag_encoded"].unique()
    
    for ind in individuals:
        # Filtramos los datos del individuo actual
        ind_data = df[df["nametag_encoded"] == ind]
    
        # Convertimos los datos del individuo en un array numpy
        ind_data_np = ind_data.to_numpy()
        
        # Como ahora incluimos también el valor de clima de modelos del momento actual hay que cambiar para window_size + 1
        for i in range(len(ind_data_np) - (window_size + 1)):
            row = []
            for j in range(i, i + window_size + 1):
                if j < i + window_size:  # Para las ventanas pasadas (t-3, t-2, t-1)
                    # Excluir las columnas del modelo
                    row.append([value for k, value in enumerate(ind_data_np[j]) if k not in model_indices])
                else:  # Para el tiempo actual (t)
                    # Incluir solo las columnas del modelo
                    current_time_values = [value for k, value in enumerate(ind_data_np[j]) if k in model_indices]

                    # Con esta linea hacemos que se metan las columnas con descriptores en el current year
                    current_descriptor_values = [value for k, value in enumerate(ind_data_np[j]) if k in descriptor_indices]

                    # Rellenar con ceros para que tenga la misma longitud que las ventanas pasadas
                    filled_current_time_values = current_descriptor_values + [0] * (len(non_model_indices) - (len(current_time_values) + len(current_descriptor_values))) + current_time_values
                    row.append(filled_current_time_values)
                    
            X.append(row)
            label = ind_data_np[i + window_size][bai_index]  # Cogemos solo [bai_index] porque queremos predecir SOLO la primera columna que es "bai"
            y.append(label)  # Eso se sigue quedando igual porque solo se predice t no el resto de variables
    
    return np.array(X), np.array(y)
    

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
    
def dataStructureSimulatedClimate(df):
    # Identificamos las columnas que contienen '_model'
    model_cols = [col for col in df.columns if '_model' in col]

    # Grupo por 'nametag'
    grouped = df.groupby('nametag_encoded')

    # Lista para almacenar los DataFrames procesados
    processed_dfs = []

    for name, group in grouped:
                
        # Desplazamos las columnas especificadas
        group[model_cols] = group[model_cols].shift(-1)

        # Eliminamos la última fila (la que tendrá NA tras el shift)
        group = group.dropna(subset=model_cols)

        # Añadimos el DataFrame procesado a la lista
        processed_dfs.append(group)

        # Concatenamos los DataFrames procesados
        result_df = pd.concat(processed_dfs)

    return result_df