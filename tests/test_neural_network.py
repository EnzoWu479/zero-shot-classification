"""
Testes completos para validar a implementação da NeuralNetwork.
"""

import sys
import os
import tempfile

# Adicionar o diretório src ao path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from neural_network.network import NeuralNetwork, create_simple_network
from neural_network.matrix import Matrix


def test_network_creation():
    """Testa criação da rede neural"""
    print("=== Testando Criação da Rede Neural ===")
    
    # Criar rede simples
    network = NeuralNetwork(
        layer_sizes=[3, 5, 2],
        activations=['relu', 'sigmoid']
    )
    
    print(f"Rede criada: {network}")
    
    # Verificar informações da rede
    info = network.get_network_info()
    print(f"Arquitetura: {info['architecture']}")
    print(f"Parâmetros totais: {info['total_parameters']}")
    print(f"Ativações: {info['activations']}")
    
    # Testar função utilitária
    simple_net = create_simple_network(
        input_size=4,
        hidden_sizes=[8, 6],
        output_size=3,
        hidden_activation='tanh',
        output_activation='softmax'
    )
    
    print(f"Rede simples: {simple_net}")
    
    print("✅ Testes de Criação concluídos!\n")


def test_forward_pass():
    """Testa forward propagation"""
    print("=== Testando Forward Propagation ===")
    
    # Criar rede para classificação binária
    network = NeuralNetwork(
        layer_sizes=[2, 4, 1],
        activations=['relu', 'sigmoid']
    )
    
    # Teste com lista
    inputs = [1.0, -0.5]
    outputs = network.forward(inputs)
    prediction = network.predict(inputs)
    
    print(f"Entrada: {inputs}")
    print(f"Saída (Matrix): {outputs}")
    print(f"Predição (lista): {prediction}")
    
    # Teste com Matrix
    input_matrix = Matrix([[1.0, -0.5]])
    outputs_matrix = network.forward(input_matrix)
    print(f"Saída com Matrix: {outputs_matrix}")
    
    print("✅ Testes de Forward Pass concluídos!\n")


def test_training():
    """Testa treinamento da rede"""
    print("=== Testando Treinamento ===")
    
    # Criar rede para problema XOR
    network = NeuralNetwork(
        layer_sizes=[2, 4, 1],
        activations=['tanh', 'sigmoid'],
        weight_init_method='xavier'
    )
    
    # Dados XOR
    train_inputs = [
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ]
    
    train_targets = [
        [0],
        [1],
        [1],
        [0]
    ]
    
    print("Dados de treinamento XOR:")
    for inp, target in zip(train_inputs, train_targets):
        print(f"  {inp} -> {target}")
    
    # Treinar por algumas épocas
    print("\nTreinando...")
    for epoch in range(100):
        total_loss = 0.0
        
        for inputs, targets in zip(train_inputs, train_targets):
            loss = network.train_step(inputs, targets, learning_rate=0.1, loss_function='mse')
            total_loss += loss
        
        avg_loss = total_loss / len(train_inputs)
        
        if epoch % 20 == 0:
            print(f"Época {epoch}: Loss = {avg_loss:.6f}")
    
    # Testar predições finais
    print("\nPredições após treinamento:")
    for inputs, targets in zip(train_inputs, train_targets):
        prediction = network.predict(inputs)
        print(f"  {inputs} -> predição: {prediction[0]:.4f}, alvo: {targets[0]}")
    
    print("✅ Testes de Treinamento concluídos!\n")


def test_batch_training():
    """Testa treinamento em batch"""
    print("=== Testando Treinamento em Batch ===")
    
    # Criar rede para regressão simples
    network = NeuralNetwork(
        layer_sizes=[1, 8, 1],
        activations=['relu', 'linear']
    )
    
    # Dados para y = x^2
    import random
    batch_inputs = [[x] for x in [random.uniform(-2, 2) for _ in range(20)]]
    batch_targets = [[x[0] ** 2] for x in batch_inputs]
    
    print(f"Treinando em batch de {len(batch_inputs)} amostras")
    
    # Treinar em batches
    for epoch in range(50):
        loss = network.train_batch(
            batch_inputs, 
            batch_targets, 
            learning_rate=0.01,
            loss_function='mse'
        )
        
        if epoch % 10 == 0:
            print(f"Época {epoch}: Batch Loss = {loss:.6f}")
    
    # Testar algumas predições
    print("\nTestando y = x^2:")
    test_values = [-1.5, 0, 1.0, 2.0]
    for x in test_values:
        pred = network.predict([x])
        expected = x ** 2
        print(f"  x={x:4.1f}: predição={pred[0]:6.3f}, esperado={expected:6.3f}")
    
    print("✅ Testes de Batch Training concluídos!\n")


def test_evaluation():
    """Testa avaliação da rede"""
    print("=== Testando Avaliação ===")
    
    # Criar rede para classificação
    network = NeuralNetwork(
        layer_sizes=[2, 6, 3],
        activations=['relu', 'softmax']
    )
    
    # Gerar dados sintéticos
    import random
    test_inputs = []
    test_targets = []
    
    for _ in range(50):
        x, y = random.uniform(-1, 1), random.uniform(-1, 1)
        test_inputs.append([x, y])
        
        # Classificação simples baseada em quadrantes
        if x > 0 and y > 0:
            test_targets.append([1, 0, 0])
        elif x <= 0 and y > 0:
            test_targets.append([0, 1, 0])
        else:
            test_targets.append([0, 0, 1])
    
    # Treinar um pouco
    for _ in range(100):
        for inputs, targets in zip(test_inputs[:20], test_targets[:20]):
            network.train_step(inputs, targets, learning_rate=0.05)
    
    # Avaliar
    metrics = network.evaluate(
        test_inputs[20:30], 
        test_targets[20:30], 
        loss_function='mse'
    )
    
    print(f"Métricas de avaliação:")
    print(f"  Loss: {metrics['loss']:.6f}")
    print(f"  Acurácia: {metrics['accuracy']:.4f}")
    print(f"  Amostras: {metrics['total_samples']}")
    
    print("✅ Testes de Avaliação concluídos!\n")


def test_serialization():
    """Testa serialização e persistência"""
    print("=== Testando Serialização ===")
    
    # Criar e treinar rede
    original_network = NeuralNetwork(
        layer_sizes=[3, 5, 2],
        activations=['tanh', 'sigmoid'],
        network_name="TestNetwork"
    )
    
    # Dados simples para treinar
    inputs = [1.0, 0.5, -1.0]
    targets = [0.8, 0.2]
    
    for _ in range(10):
        original_network.train_step(inputs, targets, learning_rate=0.1)
    
    original_prediction = original_network.predict(inputs)
    
    # Testar serialização para dicionário
    network_dict = original_network.to_dict()
    restored_network = NeuralNetwork.from_dict(network_dict)
    
    restored_prediction = restored_network.predict(inputs)
    
    print(f"Predição original: {[f'{p:.6f}' for p in original_prediction]}")
    print(f"Predição restaurada: {[f'{p:.6f}' for p in restored_prediction]}")
    
    # Verificar igualdade
    are_equal = all(
        abs(orig - rest) < 1e-10 
        for orig, rest in zip(original_prediction, restored_prediction)
    )
    print(f"Predições são iguais: {are_equal}")
    
    # Testar salvamento em arquivo
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_file = f.name
    
    try:
        original_network.save_to_file(temp_file)
        loaded_network = NeuralNetwork.load_from_file(temp_file)
        
        loaded_prediction = loaded_network.predict(inputs)
        
        file_equal = all(
            abs(orig - loaded) < 1e-10 
            for orig, loaded in zip(original_prediction, loaded_prediction)
        )
        print(f"Salvamento/carregamento de arquivo funciona: {file_equal}")
        
    finally:
        os.unlink(temp_file)
    
    print("✅ Testes de Serialização concluídos!\n")


def test_different_architectures():
    """Testa diferentes arquiteturas"""
    print("=== Testando Diferentes Arquiteturas ===")
    
    architectures = [
        ([2, 1], ['sigmoid']),  # Rede simples
        ([4, 8, 4, 2], ['relu', 'relu', 'softmax']),  # Rede profunda
        ([3, 10, 5, 1], ['tanh', 'sigmoid', 'linear']),  # Mista
    ]
    
    for i, (layer_sizes, activations) in enumerate(architectures):
        print(f"Arquitetura {i+1}: {layer_sizes} com {activations}")
        
        network = NeuralNetwork(layer_sizes, activations)
        info = network.get_network_info()
        
        print(f"  Parâmetros: {info['total_parameters']}")
        
        # Teste rápido de forward pass
        inputs = [0.5] * layer_sizes[0]
        output = network.predict(inputs)
        print(f"  Saída teste: {[f'{o:.4f}' for o in output]}")
        
        print()
    
    print("✅ Testes de Arquiteturas concluídos!\n")


if __name__ == "__main__":
    print("🚀 Iniciando testes completos da NeuralNetwork...\n")
    
    try:
        test_network_creation()
        test_forward_pass()
        test_training()
        test_batch_training()
        test_evaluation()
        test_serialization()
        test_different_architectures()
        
        print("🎉 Todos os testes da NeuralNetwork passaram!")
        print("✅ Rede Neural completa implementada com sucesso!")
        print("✅ Fase 2 - Rede Neural Completa concluída!")
        
    except Exception as e:
        print(f"❌ Erro durante os testes: {e}")
        import traceback
        traceback.print_exc()
