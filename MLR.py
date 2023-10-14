#Hecho por Alexandro Gutierrez Serna

import pandas as pd
import numpy as np

# Cargar los datos desde el archivo CSV
data = pd.read_csv("Student_Performance.csv")

# Convertir la columna 'Extracurricular Activities' a números enteros
data['Extracurricular Activities'] = data['Extracurricular Activities'].apply(lambda x: 1 if x == 'Yes' else 0)

# Dividir los datos en características (X) y variable objetivo (y)
X = data[["Hours Studied", "Previous Scores", "Extracurricular Activities", "Sleep Hours", "Sample Question Papers Practiced"]]
y = data["Performance Index"]

# Dividir los datos en conjuntos de entrenamiento y prueba (70% entrenamiento, 30% prueba)
train_size = int(0.7 * len(data))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Calcular manualmente el modelo de regresión lineal
X_train = np.c_[np.ones(X_train.shape[0]), X_train]  # Agrega columna de unos para el sesgo (intercept)
X_test = np.c_[np.ones(X_test.shape[0]), X_test]


# Calcular los coeficientes de regresión mediante la ecuación normal

#X_traon.T es la transpuesta de X_train
#X_train.T @ X_train es la matriz de covarianza
#np.linalg.inv(X_train.T @ X_train) es la inversa de la matriz de covarianza
#X_train.T @ y_train es la matriz de covarianza por el vector objetivo
#Se multiplica la inversa de la matriz de covarianza por la matriz de covarianza por el vector objetivo

#Finalmente resulta en un vector de coeficientes
coefficients = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ y_train

print("X_test:", X_test)
print("coefficients:", coefficients)

# Realizar predicciones en el conjunto de prueba
y_pred = X_test @ coefficients

# Crear un DataFrame con las predicciones y los valores reales de y
y_pred_df = pd.DataFrame({'Predicted Values': y_pred, 'Actual Values': y_test})

# Guardar el DataFrame en un archivo CSV
y_pred_df.to_csv('predicciones.csv', index=False)

# Calcular el error cuadrático medio (MSE) de las predicciones
mse = ((y_test - y_pred) ** 2).mean()
print(f"Error cuadrático medio: {mse}")

# Tomar los primeros 10 registros de X_train y excluir la primera columna en cada registro
nuevos_datos = X_test[:10, 1:]

#Se realiza la prediccion multiplicando la matriz de nuevos datos con los coeficientes
#Donde coefficients[0] es el intercepto y coefficients[1:] son los coeficientes de las caracteristicas
prediccion = nuevos_datos @ coefficients[1:] + coefficients[0]

print(f"Predicción de Performance Index en {nuevos_datos}: {prediccion}")