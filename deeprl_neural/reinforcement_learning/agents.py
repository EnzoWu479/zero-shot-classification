#!/usr/bin/env python3
"""
Agentes de Reinforcement Learning

Este módulo implementa diferentes tipos de agentes de RL:
- QLearningAgent: Agente clássico com Q-table
- DQNAgent: Deep Q-Network com rede neural
- AgentFactory: Factory pattern para criação de agentes

Implementado seguindo TDD (Test-Driven Development)
"""

import random
import math
from typing import List, Dict, Any, Union, Optional
from abc import ABC, abstractmethod

from ..neural_network.network import NeuralNetwork
from ..neural_network.matrix import Matrix
from .experience_replay import ExperienceReplayBuffer
from .policies import PolicyFactory


class RLAgent(ABC):
    """Classe base abstrata para agentes de RL"""
    
    def __init__(self, state_size: int, action_size: int, learning_rate: float = 0.01, 
                 discount_factor: float = 0.99):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.step_count = 0
    
    @abstractmethod
    def select_action(self, state) -> int:
        """Seleciona uma ação dado um estado"""
        pass
    
    @abstractmethod
    def update(self, state, action: int, reward: float, next_state, done: bool):
        """Atualiza o agente com uma experiência"""
        pass
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(state_size={self.state_size}, action_size={self.action_size})"


class QLearningAgent(RLAgent):
    """
    Agente Q-Learning clássico com tabela Q
    
    Implementa o algoritmo Q-Learning usando uma tabela para armazenar
    os valores Q(s,a) para cada par estado-ação.
    """
    
    def __init__(self, state_size: int, action_size: int, learning_rate: float = 0.1,
                 discount_factor: float = 0.95, exploration_policy: str = 'epsilon_greedy',
                 exploration_params: Optional[Dict] = None):
        super().__init__(state_size, action_size, learning_rate, discount_factor)
        
        # Inicializar Q-table
        self.q_table = [[0.0 for _ in range(action_size)] for _ in range(state_size)]
        
        # Configurar política de exploração
        if exploration_params is None:
            exploration_params = {}
        
        self.policy = PolicyFactory.create_policy(
            policy_type=exploration_policy,
            action_size=action_size,
            **exploration_params
        )
    
    def select_action(self, state: int) -> int:
        """Seleciona ação usando política de exploração"""
        q_values = self.q_table[state]
        return self.policy.select_action(q_values)
    
    def update(self, state: int, action: int, reward: float, next_state: int, done: bool):
        """Atualiza Q-table usando equação de Bellman"""
        current_q = self.q_table[state][action]
        
        if done:
            target_q = reward
        else:
            max_next_q = max(self.q_table[next_state])
            target_q = reward + self.discount_factor * max_next_q
        
        # Atualização Q-Learning
        self.q_table[state][action] = current_q + self.learning_rate * (target_q - current_q)
        
        # Atualizar política de exploração
        self.policy.update()
        self.step_count += 1
    
    def get_q_values(self, state: int) -> List[float]:
        """Retorna Q-values para um estado"""
        return self.q_table[state].copy()
    
    def get_best_action(self, state: int) -> int:
        """Retorna a melhor ação (greedy) para um estado"""
        q_values = self.q_table[state]
        return q_values.index(max(q_values))


class DQNAgent(RLAgent):
    """
    Deep Q-Network Agent
    
    Implementa DQN usando uma rede neural para aproximar a função Q.
    Inclui experience replay e target network para estabilidade.
    """
    
    def __init__(self, network: NeuralNetwork, learning_rate: float = 0.001,
                 discount_factor: float = 0.99, exploration_policy: str = 'epsilon_greedy',
                 exploration_params: Optional[Dict] = None, replay_buffer_size: int = 10000,
                 batch_size: int = 32, target_update_frequency: int = 1000):
        
        super().__init__(
            state_size=network.layer_sizes[0],
            action_size=network.layer_sizes[-1],
            learning_rate=learning_rate,
            discount_factor=discount_factor
        )
        
        # Redes neural principal e target
        self.network = network
        self.target_network = NeuralNetwork(
            layer_sizes=network.layer_sizes,
            activations=network.activations,
            network_name="Target_" + network.network_name
        )
        self._update_target_network()
        
        # Experience replay
        self.replay_buffer = ExperienceReplayBuffer(max_size=replay_buffer_size)
        self.batch_size = batch_size
        
        # Política de exploração
        if exploration_params is None:
            exploration_params = {}
        
        self.policy = PolicyFactory.create_policy(
            policy_type=exploration_policy,
            action_size=self.action_size,
            **exploration_params
        )
        
        # Configurações de treinamento
        self.target_update_frequency = target_update_frequency
        self.last_target_update = 0
    
    def get_q_values(self, state: Union[List, list]) -> List[float]:
        """Calcula Q-values usando a rede neural"""
        if isinstance(state, list):
            state_matrix = Matrix([state])
        else:
            state_matrix = Matrix([state.tolist()])
        
        output = self.network.forward(state_matrix)
        # A rede retorna uma matriz onde cada linha é um neurônio de saída
        # Extrair o primeiro (e único) valor de cada linha
        return [row[0] for row in output.data]
    
    def select_action(self, state: Union[List, list]) -> int:
        """Seleciona ação usando política de exploração"""
        q_values = self.get_q_values(state)
        return self.policy.select_action(q_values)
    
    def store_experience(self, state, action: int, reward: float, next_state, done: bool):
        """Armazena experiência no buffer de replay"""
        from .experience_replay import Experience
        experience = Experience(state, action, reward, next_state, done)
        self.replay_buffer.add_experience(experience)
    
    def can_train(self) -> bool:
        """Verifica se há experiências suficientes para treinar"""
        return len(self.replay_buffer) >= self.batch_size
    
    def train_step(self) -> float:
        """Executa um passo de treinamento da rede"""
        if not self.can_train():
            return 0.0
        
        # Amostrar batch de experiências
        experiences = self.replay_buffer.sample(self.batch_size)
        
        # Preparar dados para treinamento
        states = []
        targets = []
        
        for exp in experiences:
            state = exp.state if isinstance(exp.state, list) else exp.state.tolist()
            next_state = exp.next_state if isinstance(exp.next_state, list) else exp.next_state.tolist()
            
            # Q-values atuais
            current_q_values = self.get_q_values(state)
            
            # Calcular target
            if exp.done:
                target_q = exp.reward
            else:
                next_q_values = self._get_target_q_values(next_state)
                target_q = exp.reward + self.discount_factor * max(next_q_values)
            
            # Atualizar Q-value da ação tomada
            target_q_values = current_q_values.copy()
            target_q_values[exp.action] = target_q
            
            states.append(state)
            targets.append(target_q_values)
        
        # Treinar rede neural usando train_batch
        loss = self.network.train_batch(states, targets, self.learning_rate)
        
        # Atualizar política de exploração
        self.policy.update()
        self.step_count += 1
        
        return loss
    
    def _get_target_q_values(self, state: List) -> List[float]:
        """Calcula Q-values usando a target network"""
        state_matrix = Matrix([state])
        output = self.target_network.forward(state_matrix)
        return output.data[0]
    
    def update_target_network(self) -> bool:
        """Atualiza target network se necessário"""
        if self.step_count - self.last_target_update >= self.target_update_frequency:
            self._update_target_network()
            self.last_target_update = self.step_count
            return True
        return False
    
    def _update_target_network(self):
        """Copia pesos da rede principal para target network"""
        # Salvar rede principal e carregar na target
        main_state = self.network.to_dict()
        self.target_network.from_dict(main_state)
    
    def update(self, state, action: int, reward: float, next_state, done: bool):
        """Interface compatível com RLAgent base"""
        self.store_experience(state, action, reward, next_state, done)
        if self.can_train():
            loss = self.train_step()
            self.update_target_network()
            return loss
        return 0.0


class AgentFactory:
    """Factory para criação de agentes de RL"""
    
    @staticmethod
    def create_agent(agent_type: str, state_size: int, action_size: int, 
                    **kwargs) -> RLAgent:
        """
        Cria um agente de RL do tipo especificado
        
        Args:
            agent_type: 'qlearning' ou 'dqn'
            state_size: Tamanho do espaço de estados
            action_size: Tamanho do espaço de ações
            **kwargs: Parâmetros específicos do agente
        """
        if agent_type.lower() == 'qlearning':
            return AgentFactory._create_qlearning_agent(state_size, action_size, **kwargs)
        elif agent_type.lower() == 'dqn':
            return AgentFactory._create_dqn_agent(state_size, action_size, **kwargs)
        else:
            raise ValueError(f"Tipo de agente não suportado: {agent_type}")
    
    @staticmethod
    def _create_qlearning_agent(state_size: int, action_size: int, **kwargs) -> QLearningAgent:
        """Cria agente Q-Learning"""
        return QLearningAgent(
            state_size=state_size,
            action_size=action_size,
            learning_rate=kwargs.get('learning_rate', 0.1),
            discount_factor=kwargs.get('discount_factor', 0.95),
            exploration_policy=kwargs.get('exploration_policy', 'epsilon_greedy'),
            exploration_params=kwargs.get('exploration_params', {})
        )
    
    @staticmethod
    def _create_dqn_agent(state_size: int, action_size: int, **kwargs) -> DQNAgent:
        """Cria agente DQN"""
        # Criar arquitetura da rede neural
        hidden_layers = kwargs.get('hidden_layers', [64, 32])
        architecture = [state_size] + hidden_layers + [action_size]
        
        # Ativações padrão
        activations = ['relu'] * (len(architecture) - 2) + ['linear']
        
        # Criar rede neural
        network = NeuralNetwork(
            layer_sizes=architecture,
            activations=activations,
            network_name=kwargs.get('network_name', 'DQN')
        )
        
        return DQNAgent(
            network=network,
            learning_rate=kwargs.get('learning_rate', 0.001),
            discount_factor=kwargs.get('discount_factor', 0.99),
            exploration_policy=kwargs.get('exploration_policy', 'epsilon_greedy'),
            exploration_params=kwargs.get('exploration_params', {}),
            replay_buffer_size=kwargs.get('replay_buffer_size', 10000),
            batch_size=kwargs.get('batch_size', 32),
            target_update_frequency=kwargs.get('target_update_frequency', 1000)
        )
    
    @staticmethod
    def get_available_types() -> List[str]:
        """Retorna tipos de agentes disponíveis"""
        return ['qlearning', 'dqn']


# Função de conveniência para testes
def create_simple_qlearning_agent(state_size: int = 10, action_size: int = 4) -> QLearningAgent:
    """Cria um agente Q-Learning simples para testes"""
    return AgentFactory.create_agent('qlearning', state_size, action_size)


def create_simple_dqn_agent(state_size: int = 4, action_size: int = 2) -> DQNAgent:
    """Cria um agente DQN simples para testes"""
    return AgentFactory.create_agent('dqn', state_size, action_size, hidden_layers=[16, 8])
