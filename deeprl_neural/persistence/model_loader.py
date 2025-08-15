#!/usr/bin/env python3
"""
Model Loader - Sistema de carregamento de modelos de rede neural
Implementação em Python puro para deserialização de redes neurais e agentes RL
"""

import json
import os
from typing import Dict, Any, Union, Optional

from ..neural_network.network import NeuralNetwork
from ..neural_network.layer import Layer
from ..neural_network.neuron import Neuron
from ..neural_network.activation import ActivationFunction
from ..reinforcement_learning.agents import QLearningAgent, DQNAgent
from ..reinforcement_learning.experience_replay import ExperienceReplayBuffer, Experience
from ..reinforcement_learning.policies import (
    EpsilonGreedyPolicy, BoltzmannPolicy, UCBPolicy,
    RandomPolicy, GreedyPolicy, ThompsonSamplingPolicy
)


class ModelLoader:
    """
    Classe responsável pelo carregamento de modelos de rede neural e agentes RL
    """
    
    def __init__(self, base_path: str = "models"):
        """
        Inicializa o ModelLoader
        
        Args:
            base_path: Diretório base dos modelos salvos
        """
        self.base_path = base_path
    
    def load_neural_network(self, filename: str) -> tuple[NeuralNetwork, Dict[str, Any]]:
        """
        Carrega uma rede neural de arquivo JSON
        
        Args:
            filename: Nome do arquivo (com ou sem extensão)
            
        Returns:
            Tupla com (rede_neural, metadados)
        """
        filepath = self._get_full_path(filename)
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if data.get('type') != 'neural_network':
            raise ValueError(f"Arquivo não contém uma rede neural: {data.get('type')}")
        
        network_data = data['network']
        metadata = data.get('metadata', {})
        
        # Reconstruir rede neural
        network = self._build_neural_network_from_data(network_data)
        
        return network, metadata
    
    def load_qlearning_agent(self, filename: str) -> tuple[QLearningAgent, Dict[str, Any]]:
        """
        Carrega um agente Q-Learning
        
        Args:
            filename: Nome do arquivo (com ou sem extensão)
            
        Returns:
            Tupla com (agente, metadados)
        """
        filepath = self._get_full_path(filename)
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if data.get('type') != 'qlearning_agent':
            raise ValueError(f"Arquivo não contém um agente Q-Learning: {data.get('type')}")
        
        agent_data = data['agent']
        metadata = data.get('metadata', {})
        
        # Reconstruir agente
        agent = self._build_qlearning_agent_from_data(agent_data)
        
        return agent, metadata
    
    def load_dqn_agent(self, filename: str) -> tuple[DQNAgent, Dict[str, Any]]:
        """
        Carrega um agente DQN
        
        Args:
            filename: Nome do arquivo (com ou sem extensão)
            
        Returns:
            Tupla com (agente, metadados)
        """
        filepath = self._get_full_path(filename)
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if data.get('type') != 'dqn_agent':
            raise ValueError(f"Arquivo não contém um agente DQN: {data.get('type')}")
        
        agent_data = data['agent']
        metadata = data.get('metadata', {})
        
        # Reconstruir agente
        agent = self._build_dqn_agent_from_data(agent_data)
        
        return agent, metadata
    
    def load_training_history(self, session_name: str) -> Dict[str, Any]:
        """
        Carrega o histórico de treinamento de uma sessão
        
        Args:
            session_name: Nome da sessão de treinamento
            
        Returns:
            Dados do histórico de treinamento
        """
        history_path = os.path.join(self.base_path, session_name, "training_history.json")
        
        with open(history_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return data
    
    def load_model_auto(self, filename: str) -> tuple[Any, Dict[str, Any], str]:
        """
        Carrega um modelo automaticamente detectando o tipo
        
        Args:
            filename: Nome do arquivo
            
        Returns:
            Tupla com (modelo, metadados, tipo)
        """
        filepath = self._get_full_path(filename)
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        model_type = data.get('type')
        
        if model_type == 'neural_network':
            model, metadata = self.load_neural_network(filename)
            return model, metadata, 'neural_network'
        elif model_type == 'qlearning_agent':
            model, metadata = self.load_qlearning_agent(filename)
            return model, metadata, 'qlearning_agent'
        elif model_type == 'dqn_agent':
            model, metadata = self.load_dqn_agent(filename)
            return model, metadata, 'dqn_agent'
        else:
            raise ValueError(f"Tipo de modelo não suportado: {model_type}")
    
    def _get_full_path(self, filename: str) -> str:
        """Constrói o caminho completo do arquivo"""
        if not filename.endswith('.json'):
            filename += '.json'
        
        if os.path.isabs(filename):
            return filename
        
        return os.path.join(self.base_path, filename)
    
    def _build_neural_network_from_data(self, network_data: Dict[str, Any]) -> NeuralNetwork:
        """Reconstrói uma rede neural a partir dos dados"""
        layer_sizes = network_data['layer_sizes']
        activations = network_data['activations']
        
        # Criar rede
        network = NeuralNetwork(layer_sizes, activations)
        
        # Restaurar pesos e biases
        layers_data = network_data['layers']
        for i, layer_data in enumerate(layers_data):
            layer = network.layers[i]
            neurons_data = layer_data['neurons']
            
            for j, neuron_data in enumerate(neurons_data):
                neuron = layer.neurons[j]
                neuron.weights = neuron_data['weights'].copy()
                neuron.bias = neuron_data['bias']
        
        return network
    
    def _build_qlearning_agent_from_data(self, agent_data: Dict[str, Any]) -> QLearningAgent:
        """Reconstrói um agente Q-Learning a partir dos dados"""
        # Criar agente base
        agent = QLearningAgent(
            state_size=agent_data['state_size'],
            action_size=agent_data['action_size'],
            learning_rate=agent_data['learning_rate'],
            discount_factor=agent_data['discount_factor']
        )
        
        # Restaurar Q-table
        agent.q_table = agent_data['q_table'].copy()
        agent.step_count = agent_data['step_count']
        
        # Restaurar política
        policy_type = agent_data['policy_type']
        policy_state = agent_data.get('policy_state', {})
        agent.policy = self._build_policy_from_data(policy_type, policy_state, agent_data['action_size'])
        
        return agent
    
    def _build_dqn_agent_from_data(self, agent_data: Dict[str, Any]) -> DQNAgent:
        """Reconstrói um agente DQN a partir dos dados"""
        # Criar redes neurais
        main_network = self._build_neural_network_from_data(agent_data['main_network'])
        target_network = self._build_neural_network_from_data(agent_data['target_network'])
        
        # Criar agente
        agent = DQNAgent(
            state_size=agent_data['state_size'],
            action_size=agent_data['action_size'],
            learning_rate=agent_data['learning_rate'],
            discount_factor=agent_data['discount_factor'],
            batch_size=agent_data.get('batch_size', 32),
            target_update_frequency=agent_data.get('target_update_frequency', 100)
        )
        
        # Substituir redes
        agent.network = main_network
        agent.target_network = target_network
        
        # Restaurar estado
        agent.step_count = agent_data['step_count']
        agent.last_target_update = agent_data.get('last_target_update', 0)
        
        # Restaurar política
        policy_type = agent_data['policy_type']
        policy_state = agent_data.get('policy_state', {})
        agent.policy = self._build_policy_from_data(policy_type, policy_state, agent_data['action_size'])
        
        # Restaurar buffer de replay
        replay_buffer_size = agent_data.get('replay_buffer_size', 10000)
        agent.replay_buffer = ExperienceReplayBuffer(replay_buffer_size)
        
        # Restaurar experiências
        experiences_data = agent_data.get('replay_buffer_experiences', [])
        for exp_data in experiences_data:
            experience = Experience(
                state=exp_data['state'],
                action=exp_data['action'],
                reward=exp_data['reward'],
                next_state=exp_data['next_state'],
                done=exp_data['done']
            )
            agent.replay_buffer.add(experience)
        
        return agent
    
    def _build_policy_from_data(self, policy_type: str, policy_state: Dict[str, Any], 
                               action_size: int):
        """Reconstrói uma política a partir dos dados"""
        if policy_type == 'EpsilonGreedyPolicy':
            policy = EpsilonGreedyPolicy(
                epsilon_start=policy_state.get('epsilon_start', 1.0),
                epsilon_end=policy_state.get('epsilon_end', 0.01),
                epsilon_decay=policy_state.get('epsilon_decay', 0.995)
            )
            policy.current_epsilon = policy_state.get('current_epsilon', policy.epsilon_start)
            return policy
        
        elif policy_type == 'BoltzmannPolicy':
            policy = BoltzmannPolicy(
                temperature_start=policy_state.get('temperature_start', 1.0),
                temperature_end=policy_state.get('temperature_end', 0.1),
                temperature_decay=policy_state.get('temperature_decay', 0.995)
            )
            policy.current_temperature = policy_state.get('current_temperature', policy.temperature_start)
            return policy
        
        elif policy_type == 'UCBPolicy':
            policy = UCBPolicy(
                action_size=action_size,
                c=policy_state.get('c', 1.0)
            )
            policy.action_counts = policy_state.get('action_counts', [0] * action_size)
            policy.total_steps = policy_state.get('total_steps', 0)
            return policy
        
        elif policy_type == 'RandomPolicy':
            return RandomPolicy()
        
        elif policy_type == 'GreedyPolicy':
            return GreedyPolicy()
        
        elif policy_type == 'ThompsonSamplingPolicy':
            return ThompsonSamplingPolicy(action_size)
        
        else:
            raise ValueError(f"Tipo de política não suportado: {policy_type}")
    
    def list_available_models(self) -> Dict[str, list]:
        """
        Lista todos os modelos disponíveis para carregamento
        
        Returns:
            Dicionário com listas de modelos por tipo
        """
        from ..persistence.model_saver import ModelSaver
        saver = ModelSaver(self.base_path)
        return saver.list_saved_models()
    
    def get_model_info(self, filename: str) -> Dict[str, Any]:
        """
        Obtém informações sobre um modelo
        
        Args:
            filename: Nome do arquivo do modelo
            
        Returns:
            Informações do modelo
        """
        from ..persistence.model_saver import ModelSaver
        saver = ModelSaver(self.base_path)
        return saver.get_model_info(filename)


# Funções de conveniência
def load_model(filename: str, model_type: str = "auto", 
               base_path: str = "models") -> tuple[Any, Dict[str, Any]]:
    """
    Função de conveniência para carregar modelos
    
    Args:
        filename: Nome do arquivo
        model_type: Tipo do modelo ("auto", "neural_network", "qlearning", "dqn")
        base_path: Diretório base
        
    Returns:
        Tupla com (modelo, metadados)
    """
    loader = ModelLoader(base_path)
    
    if model_type == "auto":
        model, metadata, _ = loader.load_model_auto(filename)
        return model, metadata
    elif model_type == "neural_network":
        return loader.load_neural_network(filename)
    elif model_type == "qlearning":
        return loader.load_qlearning_agent(filename)
    elif model_type == "dqn":
        return loader.load_dqn_agent(filename)
    else:
        raise ValueError(f"Tipo de modelo não suportado: {model_type}")


def load_best_model(model_type: str = "auto", base_path: str = "models") -> tuple[Any, Dict[str, Any]]:
    """
    Carrega o modelo mais recente de um tipo específico
    
    Args:
        model_type: Tipo do modelo a carregar
        base_path: Diretório base
        
    Returns:
        Tupla com (modelo, metadados)
    """
    loader = ModelLoader(base_path)
    models = loader.list_available_models()
    
    # Determinar lista de modelos baseado no tipo
    if model_type == "neural_network":
        model_list = models["neural_networks"]
    elif model_type == "qlearning":
        model_list = models["qlearning_agents"]
    elif model_type == "dqn":
        model_list = models["dqn_agents"]
    elif model_type == "auto":
        # Combinar todos os tipos e pegar o mais recente
        all_models = []
        for category, model_list in models.items():
            if category != "training_sessions":
                for model in model_list:
                    model['category'] = category
                    all_models.append(model)
        model_list = all_models
    else:
        raise ValueError(f"Tipo de modelo não suportado: {model_type}")
    
    if not model_list:
        raise FileNotFoundError(f"Nenhum modelo encontrado do tipo: {model_type}")
    
    # Ordenar por timestamp e pegar o mais recente
    model_list.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
    latest_model = model_list[0]
    
    # Carregar o modelo
    filename = latest_model['filename']
    if model_type == "auto":
        model_category = latest_model['category']
        if model_category == "neural_networks":
            return loader.load_neural_network(filename)
        elif model_category == "qlearning_agents":
            return loader.load_qlearning_agent(filename)
        elif model_category == "dqn_agents":
            return loader.load_dqn_agent(filename)
    else:
        return load_model(filename, model_type, base_path)
