"""
Testes básicos para validar a implementação da classe Matrix e funções de ativação.
"""

import pytest
import sys
import os

# Adicionar o diretório deeprl_neural ao path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'deeprl_neural'))

from neural_network.matrix import Matrix, zeros, ones, identity, random_matrix, vector
from neural_network.activation import sigmoid, relu, tanh, linear, softmax


@pytest.mark.unit
@pytest.mark.matrix
def test_matrix_creation():
    """Testa criação de matrizes"""
    m1 = Matrix([[1, 2], [3, 4]])
    assert m1.rows == 2
    assert m1.cols == 2
    assert m1.data[0][0] == 1
    assert m1.data[1][1] == 4


@pytest.mark.unit
@pytest.mark.matrix
def test_matrix_addition():
    """Testa soma de matrizes"""
    m1 = Matrix([[1, 2], [3, 4]])
    m2 = Matrix([[5, 6], [7, 8]])
    
    result = m1 + m2
    
    assert result.data[0][0] == 6  # 1 + 5
    assert result.data[0][1] == 8  # 2 + 6
    assert result.data[1][0] == 10  # 3 + 7
    assert result.data[1][1] == 12  # 4 + 8


@pytest.mark.unit
@pytest.mark.matrix
def test_matrix_multiplication():
    """Testa multiplicação de matrizes"""
    m1 = Matrix([[1, 2], [3, 4]])
    m2 = Matrix([[5, 6], [7, 8]])
    
    result = m1 * m2
    
    # Multiplicação matricial: [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]]
    assert result.data[0][0] == 19  # 1*5 + 2*7 = 5 + 14 = 19
    assert result.data[0][1] == 22  # 1*6 + 2*8 = 6 + 16 = 22
    assert result.data[1][0] == 43  # 3*5 + 4*7 = 15 + 28 = 43
    assert result.data[1][1] == 50  # 3*6 + 4*8 = 18 + 32 = 50


@pytest.mark.unit
@pytest.mark.matrix
def test_matrix_transpose():
    """Testa transposição de matriz"""
    m1 = Matrix([[1, 2, 3], [4, 5, 6]])
    
    result = m1.transpose()
    
    assert result.rows == 3
    assert result.cols == 2
    assert result.data[0][0] == 1
    assert result.data[1][0] == 2
    assert result.data[2][0] == 3
    assert result.data[0][1] == 4
    assert result.data[1][1] == 5
    assert result.data[2][1] == 6


@pytest.mark.unit
@pytest.mark.matrix
def test_matrix_utilities():
    """Testa utilitários de matriz"""
    # Teste matriz identidade
    identity_3 = identity(3)
    assert identity_3.rows == 3
    assert identity_3.cols == 3
    assert identity_3.data[0][0] == 1
    assert identity_3.data[1][1] == 1
    assert identity_3.data[2][2] == 1
    assert identity_3.data[0][1] == 0
    
    # Teste matriz de zeros
    zeros_2x3 = zeros(2, 3)
    assert zeros_2x3.rows == 2
    assert zeros_2x3.cols == 3
    assert all(zeros_2x3.data[i][j] == 0 for i in range(2) for j in range(3))
    
    # Teste matriz de uns
    ones_2x2 = ones(2, 2)
    assert ones_2x2.rows == 2
    assert ones_2x2.cols == 2
    assert all(ones_2x2.data[i][j] == 1 for i in range(2) for j in range(2))


@pytest.mark.unit
@pytest.mark.matrix
def test_random_matrix():
    """Testa criação de matriz aleatória"""
    random_m = random_matrix(3, 4, -1, 1)
    
    assert random_m.rows == 3
    assert random_m.cols == 4
    
    # Verifica se todos os valores estão no intervalo [-1, 1]
    for i in range(3):
        for j in range(4):
            assert -1 <= random_m.data[i][j] <= 1


@pytest.mark.unit
@pytest.mark.neural
def test_activation_functions():
    """Testa funções de ativação"""
    # Teste sigmoid
    assert abs(sigmoid(0) - 0.5) < 0.001
    assert sigmoid(-10) < 0.001  # Muito próximo de 0
    assert sigmoid(10) > 0.999   # Muito próximo de 1
    
    # Teste ReLU
    assert relu(-5) == 0
    assert relu(0) == 0
    assert relu(5) == 5
    
    # Teste tanh
    assert abs(tanh(0) - 0) < 0.001
    assert tanh(-10) < -0.99   # Muito próximo de -1
    assert tanh(10) > 0.99     # Muito próximo de 1
    
    # Teste linear
    assert linear(5) == 5
    assert linear(-3) == -3
    assert linear(0) == 0


@pytest.mark.unit
@pytest.mark.neural
def test_activation_functions_with_matrices():
    """Testa funções de ativação com matrizes"""
    test_matrix = Matrix([[-1, 0, 1], [2, -2, 0.5]])
    
    # Teste sigmoid com matriz
    sig_matrix = sigmoid(test_matrix)
    assert sig_matrix.rows == 2
    assert sig_matrix.cols == 3
    assert abs(sig_matrix.data[0][1] - 0.5) < 0.001  # sigmoid(0) = 0.5
    
    # Teste ReLU com matriz
    relu_matrix = relu(test_matrix)
    assert relu_matrix.data[0][0] == 0    # relu(-1) = 0
    assert relu_matrix.data[0][1] == 0    # relu(0) = 0
    assert relu_matrix.data[0][2] == 1    # relu(1) = 1
    assert relu_matrix.data[1][0] == 2    # relu(2) = 2
    assert relu_matrix.data[1][1] == 0    # relu(-2) = 0
    
    # Teste tanh com matriz
    tanh_matrix = tanh(test_matrix)
    assert abs(tanh_matrix.data[0][1] - 0) < 0.001  # tanh(0) = 0


@pytest.mark.unit
@pytest.mark.neural
def test_softmax_function():
    """Testa função softmax"""
    softmax_input = vector([1, 2, 3], column=False)
    softmax_output = softmax(softmax_input)
    
    # Softmax deve somar 1
    assert abs(softmax_output.sum() - 1.0) < 0.001
    
    # Todos os valores devem ser positivos
    for i in range(softmax_output.cols):
        assert softmax_output.data[0][i] > 0


@pytest.mark.unit
@pytest.mark.neural
def test_activation_derivatives():
    """Testa derivadas das funções de ativação"""
    test_val = 0.5
    
    # Teste derivada sigmoid
    sig_deriv = sigmoid.derivative(test_val)
    assert sig_deriv > 0  # Derivada deve ser positiva
    
    # Teste derivada ReLU
    relu_deriv = relu.derivative(test_val)
    assert relu_deriv == 1  # Para x > 0, derivada = 1
    
    relu_deriv_neg = relu.derivative(-0.5)
    assert relu_deriv_neg == 0  # Para x < 0, derivada = 0
    
    # Teste derivada tanh
    tanh_deriv = tanh.derivative(test_val)
    assert tanh_deriv > 0  # Derivada deve ser positiva
    
    # Teste derivada linear
    linear_deriv = linear.derivative(test_val)
    assert linear_deriv == 1  # Derivada da função linear sempre é 1


@pytest.mark.unit
@pytest.mark.matrix
def test_advanced_matrix_operations():
    """Testa operações avançadas de matriz"""
    print("=== Testando Operações Avançadas de Matrix ===")
    
    # Teste de inicializações especiais
    xavier_matrix = Matrix(rows=3, cols=4).xavier_init()
    print(f"Inicialização Xavier (3x4):\n{xavier_matrix}")
    
    he_matrix = Matrix(rows=3, cols=4).he_init()
    print(f"Inicialização He (3x4):\n{he_matrix}")
    
    # Teste de operações elemento a elemento
    m1 = Matrix([[1, 2], [3, 4]])
    m2 = Matrix([[2, 3], [4, 5]])
    
    element_wise = m1.element_wise_multiply(m2)
    print(f"Multiplicação elemento a elemento:\n{element_wise}")
    
    # Teste de estatísticas
    print(f"Soma total de m1: {m1.sum()}")
    print(f"Média de m1: {m1.mean():.4f}")
    print(f"Max de m1: {m1.max()}")
    print(f"Min de m1: {m1.min()}")
    
    print("✅ Testes Avançados concluídos!\n")

    """Testa operações avançadas de matriz"""
    m1 = Matrix([[1, 2], [3, 4]])
    m2 = Matrix([[2, 3], [4, 5]])
    
    # Teste multiplicação elemento a elemento
    element_wise = m1.element_wise_multiply(m2)
    assert element_wise.data[0][0] == 2   # 1 * 2
    assert element_wise.data[0][1] == 6   # 2 * 3
    assert element_wise.data[1][0] == 12  # 3 * 4
    assert element_wise.data[1][1] == 20  # 4 * 5
    
    # Teste estatísticas
    assert m1.sum() == 10    # 1 + 2 + 3 + 4
    assert m1.mean() == 2.5  # 10 / 4
    assert m1.max() == 4
    assert m1.min() == 1


# Compatibilidade com execução direta (se necessário)
if __name__ == "__main__":
    pytest.main([__file__])
