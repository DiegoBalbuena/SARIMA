import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Especifica el nombre de tu archivo Excel
archivo_excel = 'C:\\Users\\diego\\OneDrive\\Escritorio\\Modelo Matematico\\mosquito data_Chen2023.xlsx'


# Lee la segunda página (hoja) del archivo Excel en un DataFrame de pandas
segunda_pagina = pd.read_excel(archivo_excel, sheet_name=1)

# Ahora puedes trabajar con los datos en la variable "segunda_pagina"

# Crear un nuevo DataFrame que contiene solo las columnas 1 y 2
columnas_1_y_2 = segunda_pagina.iloc[1:, 0:2]

# Convertir el nuevo DataFrame en un array de Python
data = columnas_1_y_2.to_numpy()

df = pd.DataFrame(data, columns=['Dias', 'Mosquitos'])

# Dividir los datos en entrenamiento y prueba (80% entrenamiento, 20% prueba)
train_size = 0.8  # Porcentaje de datos para entrenamiento
X = df.index
y = df['Mosquitos']

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, shuffle=False)
# # Ajustar un modelo SARIMA al conjunto de entrenamiento
# model = SARIMAX(y_train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
# results = model.fit()

# # Realizar predicciones en el conjunto de prueba
# forecast_steps = len(y_test)
# forecast = results.get_forecast(steps=forecast_steps)
# forecast_mean = forecast.predicted_mean





# X = data
# size = int(len(X) * 0.8)
# train, test = X[0:size], X[size:len(X)]
# history = [x for x in train]
# predictions = list()
# for t in range(len(test)):
# 	model = ARIMA(history, order=(5,1,0))
# 	model_fit = model.fit()
# 	output = model_fit.forecast()
# 	yhat = output[0]
# 	predictions.append(yhat)
# 	obs = test[t]
# 	history.append(obs)
# 	print('predicted=%f, expected=%f' % (yhat, obs))
# error = mean_squared_error(test, predictions)
# print('Test MSE: %.3f' % error)