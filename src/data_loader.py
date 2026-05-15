import pandas as pd
import yaml
from sklearn.model_selection import train_test_split


def load_and_preprocess_data(config):
	"""
	Carga y preprocesamiento del dataset Telco Customer Churn.

	Args: 
		config(dict): Diccionario cargado desde params.yaml
	
	Returns:
		X_train, X_test, y_train, y_test (DataFrames / Series de pandas)
	"""
 
	# Carga del archivo CSV
	path = config['data']['raw_path']
	df = pd.read_csv(path)

	# Limpieza de TotalCharges: convertir a numérico e imputación a mediana
	df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors = 'coerce')
	df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

	# Eliminación de customerID
	df.drop(columns = ['customerID'], inplace = True)

	# Codificación de columnas binarias a 0 / 1
	df['gender'] = df['gender'].map({'Male': 1, 'Female': 0})
	df['Partner'] = df['Partner'].map({'Yes': 1, 'No': 0})
	df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

	# Separación de variables independientes y variable dependiente
	X = df.drop(columns = ['Churn'])
	y = df['Churn']

	# División en conjuntos de entrenamiento y prueba
	test_size = config['data']['test_size']
	random_state = config['data']['random_state']

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state = random_state, stratify = y)

	return X_train, X_test, y_train, y_test