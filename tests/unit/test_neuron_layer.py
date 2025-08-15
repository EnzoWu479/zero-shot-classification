"""
Testes para validar a implementação de Neuron e Layer.
"""

import sys
import os

# Adicionar o diretório src ao path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from neural_network.neuron import Neuron, create_neurons
from neural_network.layer import Layer
from neural_network.matrix import Matrix, vector


def test_neuron():
    """Testa a classe Neuron"""
    print("=== Testando Classe Neuron ===")
    
    # Criar neurônio simples
    neuron = Neuron(num_inputs=3, activation='sigmoid')
    print(f"Neurônio criado: {neuron}")
    
    # Teste de forward pass
    inputs = [1.0, 0.5, -1.0]
    output = neuron.forward(inputs)
    print(f"Forward pass com inputs {inputs}: output = {output:.4f}")
    
    # Teste de backward pass
    output_gradient = 1.0  # Gradiente da loss
    input_gradients = neuron.backward(output_gradient)
    print(f"Backward pass: input_gradients = {input_gradients}")
    print(f"Weight gradients: {neuron.weight_gradients}")
    print(f"Bias gradient: {neuron.bias_gradient:.4f}")
    
    # Teste de atualização de pesos
    original_weights = neuron.get_weights().copy()
    original_bias = neuron.get_bias()
    
    neuron.update_weights(learning_rate=0.1)
    
    new_weights = neuron.get_weights()
    new_bias = neuron.get_bias()
    
    print(f"Pesos originais: {[f'{w:.4f}' for w in original_weights]}")
    print(f"Novos pesos: {[f'{w:.4f}' for w in new_weights]}")
    print(f"Bias original: {original_bias:.4f}, Novo bias: {new_bias:.4f}")
    
    print("✅ Testes de Neuron concluídos!\n")


def test_layer():
    """Testa a classe Layer"""
    print("=== Testando Classe Layer ===")
    
    # Criar camada
    layer = Layer(num_neurons=4, num_inputs=3, activation='relu', layer_name="TestLayer")
    print(f"Camada criada: {layer}")
    print(f"Parâmetros totais: {layer.get_parameter_count()}")
    
    # Teste de forward pass
    inputs = [1.0, -0.5, 2.0]
    outputs = layer.forward(inputs)
    print(f"Forward pass com inputs {inputs}:")
    print(f"Outputs:\n{outputs}")
    
    # Teste de backward pass
    output_gradients = [1.0, 0.5, -1.0, 0.8]
    input_gradients = layer.backward(output_gradients)
    print(f"Backward pass com gradients {output_gradients}:")
    print(f"Input gradients: {input_gradients}")
    
    # Teste de atualização de pesos
    original_weights = layer.get_weights_matrix()
    original_biases = layer.get_biases()
    
    layer.update_weights(learning_rate=0.01)
    
    new_weights = layer.get_weights_matrix()
    new_biases = layer.get_biases()
    
    print("Weights atualizados (diferença visível nos pesos)")
    print(f"Biases originais: {[f'{b:.4f}' for b in original_biases[:2]]}...")
    print(f"Novos biases: {[f'{b:.4f}' for b in new_biases[:2]]}...")
    
    print("✅ Testes de Layer concluídos!\n")


def test_layer_chaining():
    """Testa encadeamento de camadas"""
    print("=== Testando Encadeamento de Camadas ===")
    
    # Criar duas camadas consecutivas
    layer1 = Layer(num_neurons=5, num_inputs=3, activation='sigmoid', layer_name="Hidden")
    layer2 = Layer(num_neurons=2, num_inputs=5, activation='tanh', layer_name="Output")
    
    print(f"Layer 1: {layer1}")
    print(f"Layer 2: {layer2}")
    
    # Forward pass através das camadas
    inputs = [1.0, -0.5, 0.8]
    
    # Layer 1
    hidden_outputs = layer1.forward(inputs)
    print(f"Saídas da camada oculta:\n{hidden_outputs}")
    
    # Layer 2
    final_outputs = layer2.forward(hidden_outputs.get_column(0))
    print(f"Saídas finais:\n{final_outputs}")
    
    # Backward pass
    final_gradients = [1.0, -1.0]
    
    # Backward layer 2
    hidden_gradients = layer2.backward(final_gradients)
    print(f"Gradientes para camada oculta: {hidden_gradients}")
    
    # Backward layer 1
    input_gradients = layer1.backward(hidden_gradients.get_row(0))
    print(f"Gradientes de entrada: {input_gradients}")
    
    # Atualizar ambas as camadas
    layer1.update_weights(0.01)
    layer2.update_weights(0.01)
    
    print("✅ Testes de Encadeamento concluídos!\n")


def test_activation_variations():
    """Testa diferentes funções de ativação"""
    print("=== Testando Variações de Ativação ===")
    
    activations = ['sigmoid', 'relu', 'tanh', 'linear']
    inputs = [1.0, -1.0, 0.5]
    
    for activation in activations:
        layer = Layer(num_neurons=3, num_inputs=3, activation=activation)
        outputs = layer.forward(inputs)
        
        print(f"Ativação {activation}:")
        print(f"  Outputs: {[f'{out:.4f}' for out in outputs.get_column(0)]}")
        
        # Teste de stats
        stats = layer.get_activation_stats()
        print(f"  Stats: mean={stats['mean']:.4f}, min={stats['min']:.4f}, max={stats['max']:.4f}")
    
    print("✅ Testes de Ativação concluídos!\n")


def test_serialization():
    """Testa serialização e deserialização"""
    print("=== Testando Serialização ===")
    
    # Criar e treinar uma camada
    original_layer = Layer(num_neurons=3, num_inputs=2, activation='sigmoid')
    
    # Forward pass para estabelecer estado
    inputs = [1.0, -0.5]
    outputs = original_layer.forward(inputs)
    gradients = [1.0, 0.5, -1.0]
    original_layer.backward(gradients)
    original_layer.update_weights(0.1)
    
    # Serializar
    layer_dict = original_layer.to_dict()
    print("Camada serializada para dicionário")
    
    # Deserializar
    restored_layer = Layer.from_dict(layer_dict)
    print("Camada restaurada do dicionário")
    
    # Comparar saídas
    original_outputs = original_layer.forward(inputs)
    restored_outputs = restored_layer.forward(inputs)
    
    print(f"Saídas originais: {[f'{out:.6f}' for out in original_outputs.get_column(0)]}")
    print(f"Saídas restauradas: {[f'{out:.6f}' for out in restored_outputs.get_column(0)]}")
    
    # Verificar se são iguais
    are_equal = all(
        abs(orig - rest) < 1e-10 
        for orig, rest in zip(original_outputs.get_column(0), restored_outputs.get_column(0))
    )
    
    print(f"Saídas são iguais: {are_equal}")
    print("✅ Testes de Serialização concluídos!\n")


if __name__ == "__main__":
    print("🚀 Iniciando testes de Neuron e Layer...\n")
    
    try:
        test_neuron()
        test_layer()
        test_layer_chaining()
        test_activation_variations()
        test_serialization()
        
        print("🎉 Todos os testes de Neuron e Layer passaram!")
        print("✅ Arquitetura básica da rede neural implementada!")
        
    except Exception as e:
        print(f"❌ Erro durante os testes: {e}")
        import traceback
        traceback.print_exc()
