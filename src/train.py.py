import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Carregar os dados processados
X_train = pd.read_csv('processed_train_data.csv').values
y_train = pd.read_csv('https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv').iloc[:, -1].values

# Definir e treinar o modelo
model = MLPClassifier(hidden_layer_sizes=(10,), max_iter=1000, random_state=1)
model.fit(X_train, y_train)

# Salvar o modelo treinado
import joblib
joblib.dump(model, 'diabetes_model.pkl')

print("Modelo treinado com sucesso!")
