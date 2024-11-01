import numpy as np

# Definição da função sigmoide e sua derivada
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Definindo as entradas e saídas
# Exemplo de dados (AND lógico)
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])

y = np.array([[0], [0], [0], [1]])  # Saída desejada

# Inicialização dos pesos com valores aleatórios
np.random.seed(0)  # Para reprodutibilidade
weights = np.random.rand(X.shape[1], 1)

# Treinamento da rede neural
num_iterations = 10000
for i in range(num_iterations):
    # Propagação para frente
    input_layer = X
    outputs = sigmoid(np.dot(input_layer, weights))
    
    # Cálculo do erro
    error = y - outputs
    
    # Cálculo do delta
    delta = error * sigmoid_derivative(outputs)
    
    # Atualização dos pesos
    weights += np.dot(input_layer.T, delta)

# Exibição dos resultados
print("Pesos finais após o treinamento:")
print(weights)

print("Saídas após o treinamento:")
print(outputs)
