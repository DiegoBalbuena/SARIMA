import pandas as pd
from datetime import datetime, timedelta
import numpy as np

# Especifica el nombre de tu archivo Excel
archivo_excel = 'C:\\Users\\diego\\OneDrive\\Escritorio\\Modelo Matematico\\mosquito data_Chen2023.xlsx'

# Lee la segunda página (hoja) del archivo Excel en un DataFrame de pandas
segunda_pagina = pd.read_excel(archivo_excel, sheet_name=1)

# Ahora puedes trabajar con los datos en la variable "segunda_pagina"

# Crear un nuevo DataFrame que contiene solo las columnas 1 y 2
columna1 = segunda_pagina.iloc[1:, 0]

# Convertir el nuevo DataFrame en un array de Python
datos = columna1.to_numpy()

# Convertir los valores de datos a enteros regulares
datos = [dato.item() for dato in datos]

# Definir la fecha de referencia (1 de enero de 2017)
fecha_referencia = datetime(2017, 1, 1)

# Convertir los números de días a fechas
fechas = [fecha_referencia + timedelta(days=d) for d in datos]

# Imprimir las fechas resultantes
for fecha in fechas:
    print(fecha.date())
print(datos)