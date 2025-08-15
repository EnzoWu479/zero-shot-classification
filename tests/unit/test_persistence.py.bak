#!/usr/bin/env python3
"""
Testes para o módulo de persistência
"""

import pytest
import os
import tempfile
import json
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
    return tempfile.mkdtemp()


@pytest.fixture
def model_saver(temp_dir):
    """Fixture para ModelSaver"""
    return ModelSaver(temp_dir)


@pytest.fixture
def sample_neural_network():
    """Fixture para rede neural de exemplo"""
    return NeuralNetwork([2, 3, 1])


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


@pytest.mark.unit
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.assertEqual(data['type'], 'neural_network')
        self.assertIn('network', data)
        self.assertIn('timestamp', data)
    
    def test_save_qlearning_agent(self):
        """Testa salvamento de agente Q-Learning"""
        # Criar agente de teste
        agent = QLearningAgent(4, 2, 0.1, 0.9)
        agent.step_count = 100
        
        # Salvar
        filepath = self.saver.save_qlearning_agent(agent, "test_qlearning")
        
        # Verificar arquivo criado
        self.assertTrue(os.path.exists(filepath))
        
        # Verificar conteúdo
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.assertEqual(data['type'], 'qlearning_agent')
        self.assertEqual(data['agent']['step_count'], 100)
    
    def test_save_dqn_agent(self):
        """Testa salvamento de agente DQN"""
        # Criar agente de teste
        agent = DQNAgent(4, 2, 0.001, 0.9)
        
        # Salvar
        filepath = self.saver.save_dqn_agent(agent, "test_dqn")
        
        # Verificar arquivo criado
        self.assertTrue(os.path.exists(filepath))
        
        # Verificar conteúdo
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.assertEqual(data['type'], 'dqn_agent')
        self.assertIn('main_network', data['agent'])
        self.assertIn('target_network', data['agent'])
    
    def test_list_saved_models(self):
        """Testa listagem de modelos salvos"""
        # Salvar alguns modelos
        network = NeuralNetwork([2, 1], ['sigmoid'])
        agent = QLearningAgent(4, 2)
        
        self.saver.save_neural_network(network, "network1")
        self.saver.save_qlearning_agent(agent, "agent1")
        
        # Listar modelos
        models = self.saver.list_saved_models()
        
        self.assertEqual(len(models['neural_networks']), 1)
        self.assertEqual(len(models['qlearning_agents']), 1)
        self.assertEqual(len(models['dqn_agents']), 0)


class TestModelLoader(unittest.TestCase):
    """Testes para ModelLoader"""
    
    def setUp(self):
        """Setup para cada teste"""
        self.temp_dir = tempfile.mkdtemp()
        self.saver = ModelSaver(self.temp_dir)
        self.loader = ModelLoader(self.temp_dir)
    
    def tearDown(self):
        """Cleanup após cada teste"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_load_neural_network(self):
        """Testa carregamento de rede neural"""
        # Criar e salvar rede
        original_network = NeuralNetwork([2, 3, 1], ['relu', 'sigmoid'])
        original_weights = original_network.layers[0].neurons[0].get_weights()
        
        filepath = self.saver.save_neural_network(original_network, "test_network")
        
        # Carregar rede
        loaded_network, metadata = self.loader.load_neural_network("test_network")
        
        # Verificar estrutura
        self.assertEqual(loaded_network.layer_sizes, [2, 3, 1])
        self.assertEqual(loaded_network.activations, ['relu', 'sigmoid'])
        
        # Verificar pesos preservados
        loaded_weights = loaded_network.layers[0].neurons[0].get_weights()
        self.assertEqual(original_weights, loaded_weights)
    
    def test_load_qlearning_agent(self):
        """Testa carregamento de agente Q-Learning"""
        # Criar e salvar agente
        original_agent = QLearningAgent(4, 2, 0.1, 0.9)
        original_agent.step_count = 150
        original_agent.q_table = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]
        
        self.saver.save_qlearning_agent(original_agent, "test_agent")
        
        # Carregar agente
        loaded_agent, metadata = self.loader.load_qlearning_agent("test_agent")
        
        # Verificar propriedades
        self.assertEqual(loaded_agent.state_size, 4)
        self.assertEqual(loaded_agent.action_size, 2)
        self.assertEqual(loaded_agent.step_count, 150)
        self.assertEqual(loaded_agent.q_table, original_agent.q_table)
    
    def test_load_model_auto(self):
        """Testa carregamento automático"""
        # Salvar diferentes tipos de modelos
        network = NeuralNetwork([2, 1], ['sigmoid'])
        agent = QLearningAgent(4, 2)
        
        network_path = self.saver.save_neural_network(network, "auto_network")
        agent_path = self.saver.save_qlearning_agent(agent, "auto_agent")
        
        # Carregar automaticamente
        loaded_network, _, net_type = self.loader.load_model_auto("auto_network")
        loaded_agent, _, agent_type = self.loader.load_model_auto("auto_agent")
        
        self.assertEqual(net_type, 'neural_network')
        self.assertEqual(agent_type, 'qlearning_agent')
        self.assertIsInstance(loaded_network, NeuralNetwork)
        self.assertIsInstance(loaded_agent, QLearningAgent)


class TestPersistenceFunctions(unittest.TestCase):
    """Testes para funções de conveniência"""
    
    def setUp(self):
        """Setup para cada teste"""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Cleanup após cada teste"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_save_and_load_model_functions(self):
        """Testa funções save_model e load_model"""
        # Criar modelo
        network = NeuralNetwork([2, 3, 1], ['relu', 'linear'])
        
        # Salvar usando função de conveniência
        filepath = save_model(network, "convenience_test", base_path=self.temp_dir)
        self.assertTrue(os.path.exists(filepath))
        
        # Carregar usando função de conveniência
        loaded_network, metadata = load_model("convenience_test", base_path=self.temp_dir)
        
        # Verificar
        self.assertIsInstance(loaded_network, NeuralNetwork)
        self.assertEqual(loaded_network.layer_sizes, [2, 3, 1])


if __name__ == '__main__':
    unittest.main()
