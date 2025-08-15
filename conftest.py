"""
Configuração global do pytest para o projeto Deep RL Neural Network
"""

import sys
import os
import pytest

# Adicionar src ao path Python para todos os testes
project_root = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

@pytest.fixture(scope="session")
def project_root():
    """Fixture que retorna o diretório raiz do projeto"""
    return os.path.dirname(os.path.abspath(__file__))

@pytest.fixture(scope="session") 
def src_path():
    """Fixture que retorna o caminho para o diretório src"""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src')

@pytest.fixture
def temp_dir(tmp_path):
    """Fixture que fornece um diretório temporário para testes"""
    return str(tmp_path)

@pytest.fixture
def sample_neural_network():
    """Fixture que cria uma rede neural simples para testes"""
    from neural_network.network import NeuralNetwork
    return NeuralNetwork([2, 3, 1])

@pytest.fixture
def sample_grid_world():
    """Fixture que cria um GridWorld simples para testes"""
    from environment.grid_world import GridWorld
    return GridWorld(size=3)

# Configurações de marcadores para pytest
pytest_plugins = []

def pytest_configure(config):
    """Configuração do pytest"""
    config.addinivalue_line("markers", "unit: Testes unitários")
    config.addinivalue_line("markers", "integration: Testes de integração")
    config.addinivalue_line("markers", "acceptance: Testes de aceitação")
    config.addinivalue_line("markers", "slow: Testes que demoram para executar")
    config.addinivalue_line("markers", "neural: Testes de redes neurais")
    config.addinivalue_line("markers", "rl: Testes de reinforcement learning")
    config.addinivalue_line("markers", "matrix: Testes de operações matriciais")

def pytest_collection_modifyitems(config, items):
    """Modifica itens coletados pelo pytest"""
    # Adiciona marcadores automáticos baseados no caminho do arquivo
    for item in items:
        # Marcadores baseados no diretório
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        elif "acceptance" in str(item.fspath):
            item.add_marker(pytest.mark.acceptance)
        
        # Marcadores baseados no nome do arquivo/teste
        if "neural" in str(item.fspath).lower() or "neural" in item.name.lower():
            item.add_marker(pytest.mark.neural)
        if "rl" in str(item.fspath).lower() or "agent" in str(item.fspath).lower():
            item.add_marker(pytest.mark.rl)
        if "matrix" in str(item.fspath).lower() or "matrix" in item.name.lower():
            item.add_marker(pytest.mark.matrix)
        if "slow" in item.name.lower() or "complete" in item.name.lower():
            item.add_marker(pytest.mark.slow)
