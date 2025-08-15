#!/usr/bin/env python3
"""
Testes para o módulo de persistência usando pytest
"""

import pytest
import os
import tempfile
import json
import shutil
from unittest.mock import Mock

# Imports do projeto
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from persistence.model_saver import ModelSaver, save_model
from persistence.model_loader import ModelLoader, load_model
from neural_network.network import NeuralNetwork
from reinforcement_learning.agents import QLearningAgent, DQNAgent


@pytest.fixture
def temp_dir():
    """Fixture para diretório temporário"""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    # Cleanup
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def model_saver(temp_dir):
    """Fixture para ModelSaver"""
    return ModelSaver(temp_dir)


@pytest.fixture
def model_loader(temp_dir):
    """Fixture para ModelLoader"""
    return ModelLoader(temp_dir)


@pytest.fixture
def sample_neural_network():
    """Fixture para rede neural de exemplo"""
    return NeuralNetwork([2, 3, 1])


# Testes do ModelSaver
@pytest.mark.unit
def test_model_saver_init(model_saver, temp_dir):
    """Testa inicialização do ModelSaver"""
    assert model_saver.base_path == temp_dir
    assert os.path.exists(temp_dir)


@pytest.mark.unit  
def test_save_neural_network(model_saver, temp_dir):
    """Testa salvamento de rede neural"""
    # Criar rede de teste
    network = NeuralNetwork([2, 3, 1], ['relu', 'sigmoid'])
    
    # Salvar
    filepath = model_saver.save_neural_network(network, "test_network")
    
    # Verificar arquivo criado
    assert os.path.exists(filepath)
    
    # Verificar conteúdo
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    assert data['type'] == 'neural_network'
    assert data['architecture'] == [2, 3, 1]
    assert data['activations'] == ['relu', 'sigmoid']


@pytest.mark.unit
def test_save_ql_agent(model_saver):
    """Testa salvamento de agente Q-Learning"""
    # Criar agente de teste
    agent = QLearningAgent(
        state_size=4,
        action_size=2,
        learning_rate=0.1,
        discount_factor=0.9
    )
    
    # Salvar
    filepath = model_saver.save_ql_agent(agent, "test_ql_agent")
    
    # Verificar arquivo criado
    assert os.path.exists(filepath)
    
    # Verificar conteúdo
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    assert data['type'] == 'ql_agent'
    assert data['state_size'] == 4
    assert data['action_size'] == 2
    assert data['learning_rate'] == 0.1
    assert data['discount_factor'] == 0.9


@pytest.mark.unit
def test_save_dqn_agent(model_saver):
    """Testa salvamento de agente DQN"""
    # Criar agente de teste
    agent = DQNAgent(
        state_size=4,
        action_size=2,
        learning_rate=0.001,
        discount_factor=0.95
    )
    
    # Salvar
    filepath = model_saver.save_dqn_agent(agent, "test_dqn_agent")
    
    # Verificar arquivo criado
    assert os.path.exists(filepath)
    
    # Verificar conteúdo
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    assert data['type'] == 'dqn_agent'
    assert data['state_size'] == 4
    assert data['action_size'] == 2


@pytest.mark.unit
def test_save_with_metadata(model_saver, sample_neural_network):
    """Testa salvamento com metadados"""
    metadata = {
        'training_epochs': 100,
        'accuracy': 0.95,
        'description': 'Test model'
    }
    
    filepath = model_saver.save_neural_network(
        sample_neural_network, 
        "test_with_metadata",
        metadata=metadata
    )
    
    # Verificar metadados salvos
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    assert data['metadata']['training_epochs'] == 100
    assert data['metadata']['accuracy'] == 0.95
    assert data['metadata']['description'] == 'Test model'


# Testes do ModelLoader
@pytest.mark.unit
def test_model_loader_init(model_loader, temp_dir):
    """Testa inicialização do ModelLoader"""
    assert model_loader.base_path == temp_dir


@pytest.mark.unit
def test_load_neural_network(model_saver, model_loader):
    """Testa carregamento de rede neural"""
    # Criar e salvar rede
    original_network = NeuralNetwork([2, 3, 1], ['relu', 'sigmoid'])
    filepath = model_saver.save_neural_network(original_network, "test_load")
    
    # Carregar rede
    loaded_network = model_loader.load_neural_network(os.path.basename(filepath))
    
    # Verificar se são equivalentes
    assert loaded_network.layers[0].size == original_network.layers[0].size
    assert loaded_network.layers[1].size == original_network.layers[1].size
    assert loaded_network.layers[2].size == original_network.layers[2].size


@pytest.mark.unit
def test_load_ql_agent(model_saver, model_loader):
    """Testa carregamento de agente Q-Learning"""
    # Criar e salvar agente
    original_agent = QLearningAgent(4, 2, 0.1, 0.9)
    filepath = model_saver.save_ql_agent(original_agent, "test_load_ql")
    
    # Carregar agente
    loaded_agent = model_loader.load_ql_agent(os.path.basename(filepath))
    
    # Verificar propriedades
    assert loaded_agent.state_size == original_agent.state_size
    assert loaded_agent.action_size == original_agent.action_size
    assert loaded_agent.learning_rate == original_agent.learning_rate
    assert loaded_agent.discount_factor == original_agent.discount_factor


@pytest.mark.unit
def test_load_dqn_agent(model_saver, model_loader):
    """Testa carregamento de agente DQN"""
    # Criar e salvar agente
    original_agent = DQNAgent(4, 2, 0.001, 0.95)
    filepath = model_saver.save_dqn_agent(original_agent, "test_load_dqn")
    
    # Carregar agente
    loaded_agent = model_loader.load_dqn_agent(os.path.basename(filepath))
    
    # Verificar propriedades
    assert loaded_agent.state_size == original_agent.state_size
    assert loaded_agent.action_size == original_agent.action_size


@pytest.mark.unit
def test_list_saved_models(model_saver, model_loader):
    """Testa listagem de modelos salvos"""
    # Salvar alguns modelos
    network = NeuralNetwork([2, 3, 1])
    agent = QLearningAgent(4, 2)
    
    model_saver.save_neural_network(network, "network1")
    model_saver.save_neural_network(network, "network2")
    model_saver.save_ql_agent(agent, "agent1")
    
    # Listar modelos
    models = model_loader.list_saved_models()
    
    # Verificar lista
    assert len(models) >= 3
    model_names = [model['name'] for model in models]
    assert any('network1' in name for name in model_names)
    assert any('network2' in name for name in model_names)
    assert any('agent1' in name for name in model_names)


# Testes das funções utilitárias
@pytest.mark.unit
def test_save_model_function(temp_dir):
    """Testa função save_model"""
    network = NeuralNetwork([2, 3, 1])
    
    filepath = save_model(network, "test_function", temp_dir)
    
    assert os.path.exists(filepath)
    
    # Verificar conteúdo
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    assert data['type'] == 'neural_network'


@pytest.mark.unit
def test_load_model_function(temp_dir):
    """Testa função load_model"""
    # Criar e salvar modelo
    network = NeuralNetwork([2, 3, 1])
    filepath = save_model(network, "test_load_function", temp_dir)
    
    # Carregar usando função
    loaded_model = load_model(os.path.basename(filepath), temp_dir)
    
    assert loaded_model is not None
    assert hasattr(loaded_model, 'layers')


@pytest.mark.unit
def test_error_handling(model_loader):
    """Testa tratamento de erros"""
    # Tentar carregar arquivo inexistente
    with pytest.raises(FileNotFoundError):
        model_loader.load_neural_network("nonexistent.json")
    
    # Tentar carregar arquivo com formato inválido
    invalid_file = os.path.join(model_loader.base_path, "invalid.json")
    with open(invalid_file, 'w') as f:
        f.write("invalid json content")
    
    with pytest.raises(json.JSONDecodeError):
        model_loader.load_neural_network("invalid.json")


if __name__ == "__main__":
    pytest.main([__file__])
