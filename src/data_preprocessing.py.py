import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Carregar o conjunto de dados
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv'
columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
data = pd.read_csv(url, names=columns)

# Preprocessamento
X = data.iloc[:, :-1].values  # Atributos
y = data.iloc[:, -1].values   # Resultado (se tem diabetes)

# Divisão dos dados em treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Normalização dos dados
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Salvar os dados processados
pd.DataFrame(X_train).to_csv('processed_train_data.csv', index=False)
pd.DataFrame(X_test).to_csv('processed_test_data.csv', index=False)
