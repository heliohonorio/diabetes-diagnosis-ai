import pandas as pd
import joblib
import numpy as np

# Carregar o modelo treinado
model = joblib.load('diabetes_model.pkl')

# Carregar os dados de teste
X_test = pd.read_csv('processed_test_data.csv').values

# Fazer previsões
y_pred = model.predict(X_test)

# Mostrar os resultados
print("Previsões:", y_pred)

# Avaliar o modelo
y_test = pd.read_csv('https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv').iloc[:, -1].values
from sklearn.metrics import accuracy_score
print("Acurácia:", accuracy_score(y_test, y_pred))
