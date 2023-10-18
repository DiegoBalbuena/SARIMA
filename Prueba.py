import pandas as pd

# Especifica el nombre de tu archivo Excel
archivo_excel = 'C:\\Users\\diego\\OneDrive\\Escritorio\\Modelo Matematico\\mosquito data_Chen2023.xlsx'


# Lee la segunda página (hoja) del archivo Excel en un DataFrame de pandas
segunda_pagina = pd.read_excel(archivo_excel, sheet_name=1)

# Ahora puedes trabajar con los datos en la variable "segunda_pagina"

# Crear un nuevo DataFrame que contiene solo las columnas 1 y 2
columnas_1_y_2 = segunda_pagina.iloc[1:, 0:2]

# Convertir el nuevo DataFrame en un array de Python
datos = columnas_1_y_2.to_numpy()

from statsmodels.tsa.arima_process import arma_generate_sample
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

X = []
for i in range(len(datos)):
	X.append(datos[i][1])

times = np.linspace(1, len(X), len(X), endpoint=True)
plt.plot(times, X, 'ko', linewidth=2, label='Datos')

plt.xlabel('Tiempo')
plt.ylabel('Casos')
plt.legend()
plt.show()

from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.stattools import adfuller
import numpy as np
import pmdarima as pm
from pmdarima.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm

# Asegúrate de que X contenga tus datos de series de tiempo

# Divide tus datos en conjuntos de entrenamiento y prueba
train, test = train_test_split(X, train_size=0.66)


# Mejor modelo ARIMA
print('Results of Dickey-Fuller Test:')
dftest = adfuller(train, autolag='AIC')
dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
for key,value in dftest[4].items():
    dfoutput['Critical Value (%s)'%key] = value
print(dfoutput)

d = 0

train = pd.Series(train)

# Create ACF plot to determine the order of AR component (p)
acf = sm.tsa.acf(train, fft=False)
plt.figure(figsize=(12, 4))
plt.stem(acf)
plt.title('ACF Plot')
plt.xlabel('Lag')
plt.ylabel('ACF Value')
plt.show()

# Find the lag at which ACF crosses significance threshold (usually 95% confidence interval)
# This is the value of p for your ARIMA model
alpha = 0.05  # Significance level
p = np.where(acf < alpha)[0][0]

# Create PACF plot to determine the order of MA component (q)
pacf = sm.tsa.pacf(train, method='ols')
plt.figure(figsize=(12, 4))
plt.stem(pacf)
plt.title('PACF Plot')
plt.xlabel('Lag')
plt.ylabel('PACF Value')
plt.show()

# Find the lag at which PACF crosses significance threshold (usually 95% confidence interval)
# This is the value of q for your ARIMA model
alpha = 0.05  # Significance level
q = np.where(np.abs(pacf) < alpha)[0][1]


history = [x for x in train]
predictions = list()
for t in range(len(test)):
	model = ARIMA(history, order=(p,d,q))
	model_fit = model.fit()
	output = model_fit.forecast()
	yhat = output[0]
	predictions.append(yhat)
	obs = test[t]
	history.append(obs)
	print('predicted=%f, expected=%f' % (yhat, obs))
error = mean_squared_error(test, predictions)
print('Test MSE: %.3f' % error)
print(datos)
