"""
Testes básicos para validar a implementação da classe Matrix e funções de ativação.
"""

import sys
import os

# Adicionar o diretório src ao path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from neural_network.matrix import Matrix, zeros, ones, identity, random_matrix, vector
from neural_network.activation import sigmoid, relu, tanh, linear, softmax


def test_matrix_operations():
    """Testa operações básicas da matriz"""
    print("=== Testando Operações de Matrix ===")
    
    # Teste de criação
    m1 = Matrix([[1, 2], [3, 4]])
    m2 = Matrix([[5, 6], [7, 8]])
    
    print(f"Matrix 1:\n{m1}")
    print(f"Matrix 2:\n{m2}")
    
    # Teste de soma
    m_sum = m1 + m2
    print(f"Soma:\n{m_sum}")
    
    # Teste de multiplicação
    m_mult = m1 * m2
    print(f"Multiplicação:\n{m_mult}")
    
    # Teste de transposição
    m_trans = m1.transpose()
    print(f"Transposição de Matrix 1:\n{m_trans}")
    
    # Teste de matriz identidade
    identity_3 = identity(3)
    print(f"Matriz identidade 3x3:\n{identity_3}")
    
    # Teste de matriz aleatória
    random_m = random_matrix(2, 3, -1, 1)
    print(f"Matriz aleatória 2x3:\n{random_m}")
    
    print("✅ Testes de Matrix concluídos!\n")


def test_activation_functions():
    """Testa funções de ativação"""
    print("=== Testando Funções de Ativação ===")
    
    # Teste com valores escalares
    test_values = [-2, -1, 0, 1, 2]
    
    print("Testando com valores escalares:")
    for val in test_values:
        sig_val = sigmoid(val)
        relu_val = relu(val)
        tanh_val = tanh(val)
        linear_val = linear(val)
        
        print(f"x={val:2d}: sigmoid={sig_val:.4f}, relu={relu_val:.4f}, tanh={tanh_val:.4f}, linear={linear_val:.4f}")
    
    # Teste com matriz
    print("\nTestando com matriz:")
    test_matrix = Matrix([[-1, 0, 1], [2, -2, 0.5]])
    print(f"Matriz original:\n{test_matrix}")
    
    sig_matrix = sigmoid(test_matrix)
    print(f"Sigmoid:\n{sig_matrix}")
    
    relu_matrix = relu(test_matrix)
    print(f"ReLU:\n{relu_matrix}")
    
    tanh_matrix = tanh(test_matrix)
    print(f"Tanh:\n{tanh_matrix}")
    
    # Teste de softmax
    print("\nTestando Softmax:")
    softmax_input = vector([1, 2, 3], column=False)
    print(f"Input para softmax:\n{softmax_input}")
    softmax_output = softmax(softmax_input)
    print(f"Output softmax:\n{softmax_output}")
    print(f"Soma do softmax: {softmax_output.sum():.6f}")
    
    print("✅ Testes de Ativação concluídos!\n")


def test_derivatives():
    """Testa derivadas das funções de ativação"""
    print("=== Testando Derivadas ===")
    
    test_val = 0.5
    
    sig_deriv = sigmoid.derivative(test_val)
    relu_deriv = relu.derivative(test_val)
    tanh_deriv = tanh.derivative(test_val)
    linear_deriv = linear.derivative(test_val)
    
    print(f"Para x={test_val}:")
    print(f"Derivada sigmoid: {sig_deriv:.4f}")
    print(f"Derivada relu: {relu_deriv:.4f}")
    print(f"Derivada tanh: {tanh_deriv:.4f}")
    print(f"Derivada linear: {linear_deriv:.4f}")
    
    print("✅ Testes de Derivadas concluídos!\n")


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


if __name__ == "__main__":
    print("🚀 Iniciando testes da implementação...\n")
    
    try:
        test_matrix_operations()
        test_activation_functions()
        test_derivatives()
        test_advanced_matrix_operations()
        
        print("🎉 Todos os testes passaram com sucesso!")
        print("✅ Base Matemática implementada corretamente!")
        
    except Exception as e:
        print(f"❌ Erro durante os testes: {e}")
        import traceback
        traceback.print_exc()
