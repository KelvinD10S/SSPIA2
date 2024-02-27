import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Función de activación (en este caso, función escalón)
def step_function(x):
    return np.where(x>=0, 1, 0)

# Clase del perceptrón
class Perceptron:
    def __init__(self, num_inputs, learning_rate=0.1, max_epochs=100):
        self.weights = np.random.rand(num_inputs + 1)  # +1 para el sesgo
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs

    def train(self, X, y):
        for epoch in range(self.max_epochs):
            for i in range(X.shape[0]):
                inputs = np.insert(X[i], 0, 1)  # Insertar 1 para el sesgo
                prediction = self.predict(inputs)
                error = y[i] - prediction
                self.weights += self.learning_rate * error * inputs

    def predict(self, inputs):
        summation = np.dot(inputs, self.weights)
        return step_function(summation)

# Función para leer datos de entrenamiento desde archivos CSV
def read_data(filename):
    data = pd.read_csv(filename, header=None)
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    return X, y

# Función para graficar los patrones y la recta que los separa
def plot_data_and_line(X, y, perceptron):
    plt.scatter(X[:, 0], X[:, 1], c=y)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')

    # Dibujar la línea de separación
    slope = -perceptron.weights[1] / perceptron.weights[2]
    intercept = -perceptron.weights[0] / perceptron.weights[2]
    x_values = np.linspace(-2, 2, 100)
    plt.plot(x_values, slope * x_values + intercept, 'r')

    plt.show()

# Lectura de datos de entrenamiento y prueba para XOR
X_train_xor, y_train_xor = read_data('XOR_trn.csv')
X_test_xor, y_test_xor = read_data('XOR_tst.csv')

# Creación y entrenamiento del perceptrón para XOR
perceptron_xor = Perceptron(num_inputs=X_train_xor.shape[1])
perceptron_xor.train(X_train_xor, y_train_xor)

# Visualización de los datos y la línea que los separa para XOR
print("Línea de separación para la compuerta XOR:")
plot_data_and_line(X_train_xor, y_train_xor, perceptron_xor)

# Lectura de datos de entrenamiento y prueba para OR
X_train_or, y_train_or = read_data('OR_trn.csv')
X_test_or, y_test_or = read_data('OR_tst.csv')

# Creación y entrenamiento del perceptrón para OR
perceptron_or = Perceptron(num_inputs=X_train_or.shape[1])
perceptron_or.train(X_train_or, y_train_or)

# Visualización de los datos y la línea que los separa para OR
print("Línea de separación para la compuerta OR:")
plot_data_and_line(X_train_or, y_train_or, perceptron_or)
