#!/usr/bin/env python3
"""
Model Saver - Sistema de salvamento de modelos de rede neural
Implementação em Python puro para serialização de redes neurais e agentes RL
"""

import json
import os
from datetime import datetime
from typing import Dict, Any, Union

from neural_network.network import NeuralNetwork
from reinforcement_learning.agents import QLearningAgent, DQNAgent


class ModelSaver:
    """
    Classe responsável pelo salvamento de modelos de rede neural e agentes RL
    """
    
    def __init__(self, base_path: str = "models"):
        """
        Inicializa o ModelSaver
        
        Args:
            base_path: Diretório base para salvamento dos modelos
        """
        self.base_path = base_path
        self._ensure_directory_exists()
    
    def _ensure_directory_exists(self):
        """Cria o diretório base se não existir"""
        if not os.path.exists(self.base_path):
            os.makedirs(self.base_path)
    
    def save_neural_network(self, network: NeuralNetwork, filename: str, 
                           metadata: Dict[str, Any] = None) -> str:
        """
        Salva uma rede neural em arquivo JSON
        
        Args:
            network: Rede neural a ser salva
            filename: Nome do arquivo (sem extensão)
            metadata: Metadados adicionais
            
        Returns:
            Caminho completo do arquivo salvo
        """
        # Preparar dados da rede
        network_data = network.to_dict()
        
        # Adicionar metadados
        save_data = {
            "type": "neural_network",
            "timestamp": datetime.now().isoformat(),
            "network": network_data,
            "metadata": metadata or {}
        }
        
        # Caminho do arquivo
        filepath = os.path.join(self.base_path, f"{filename}.json")
        
        # Salvar em JSON
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)
        
        return filepath
    
    def save_qlearning_agent(self, agent: QLearningAgent, filename: str,
                            metadata: Dict[str, Any] = None) -> str:
        """
        Salva um agente Q-Learning
        
        Args:
            agent: Agente Q-Learning a ser salvo
            filename: Nome do arquivo (sem extensão)
            metadata: Metadados adicionais
            
        Returns:
            Caminho completo do arquivo salvo
        """
        # Preparar dados do agente
        agent_data = {
            "state_size": agent.state_size,
            "action_size": agent.action_size,
            "learning_rate": agent.learning_rate,
            "discount_factor": agent.discount_factor,
            "q_table": agent.q_table,
            "step_count": agent.step_count,
            "policy_type": type(agent.policy).__name__,
            "policy_state": self._get_policy_state(agent.policy)
        }
        
        save_data = {
            "type": "qlearning_agent",
            "timestamp": datetime.now().isoformat(),
            "agent": agent_data,
            "metadata": metadata or {}
        }
        
        filepath = os.path.join(self.base_path, f"{filename}.json")
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)
        
        return filepath
    
    def save_dqn_agent(self, agent: DQNAgent, filename: str,
                      metadata: Dict[str, Any] = None) -> str:
        """
        Salva um agente DQN
        
        Args:
            agent: Agente DQN a ser salvo
            filename: Nome do arquivo (sem extensão)
            metadata: Metadados adicionais
            
        Returns:
            Caminho completo do arquivo salvo
        """
        # Preparar dados do agente
        agent_data = {
            "state_size": agent.state_size,
            "action_size": agent.action_size,
            "learning_rate": agent.learning_rate,
            "discount_factor": agent.discount_factor,
            "step_count": agent.step_count,
            "batch_size": agent.batch_size,
            "target_update_frequency": agent.target_update_frequency,
            "last_target_update": agent.last_target_update,
            "main_network": agent.network.to_dict(),
            "target_network": agent.target_network.to_dict(),
            "policy_type": type(agent.policy).__name__,
            "policy_state": self._get_policy_state(agent.policy),
            "replay_buffer_size": agent.replay_buffer.max_size,
            "replay_buffer_experiences": self._serialize_experiences(agent.replay_buffer)
        }
        
        save_data = {
            "type": "dqn_agent",
            "timestamp": datetime.now().isoformat(),
            "agent": agent_data,
            "metadata": metadata or {}
        }
        
        filepath = os.path.join(self.base_path, f"{filename}.json")
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)
        
        return filepath
    
    def _get_policy_state(self, policy) -> Dict[str, Any]:
        """Extrai estado serializável de uma política"""
        policy_state = {}
        
        # Epsilon-Greedy
        if hasattr(policy, 'current_epsilon'):
            policy_state['current_epsilon'] = policy.current_epsilon
        if hasattr(policy, 'epsilon_start'):
            policy_state['epsilon_start'] = policy.epsilon_start
        if hasattr(policy, 'epsilon_end'):
            policy_state['epsilon_end'] = policy.epsilon_end
        if hasattr(policy, 'epsilon_decay'):
            policy_state['epsilon_decay'] = policy.epsilon_decay
        
        # Boltzmann
        if hasattr(policy, 'current_temperature'):
            policy_state['current_temperature'] = policy.current_temperature
        if hasattr(policy, 'temperature_start'):
            policy_state['temperature_start'] = policy.temperature_start
        if hasattr(policy, 'temperature_end'):
            policy_state['temperature_end'] = policy.temperature_end
        if hasattr(policy, 'temperature_decay'):
            policy_state['temperature_decay'] = policy.temperature_decay
        
        # UCB
        if hasattr(policy, 'c'):
            policy_state['c'] = policy.c
        if hasattr(policy, 'action_counts'):
            policy_state['action_counts'] = policy.action_counts
        if hasattr(policy, 'total_steps'):
            policy_state['total_steps'] = policy.total_steps
        
        return policy_state
    
    def _serialize_experiences(self, replay_buffer) -> list:
        """Serializa experiências do buffer de replay"""
        experiences = []
        buffer_size = replay_buffer.size  # Atributo size
        
        for i in range(min(buffer_size, len(replay_buffer.buffer))):
            exp = replay_buffer.buffer[i]
            experiences.append({
                "state": exp.state,
                "action": exp.action,
                "reward": exp.reward,
                "next_state": exp.next_state,
                "done": exp.done
            })
        return experiences
    
    def save_training_session(self, agent, training_history: Dict[str, list],
                             session_name: str, metadata: Dict[str, Any] = None) -> str:
        """
        Salva uma sessão completa de treinamento
        
        Args:
            agent: Agente treinado
            training_history: Histórico de treinamento
            session_name: Nome da sessão
            metadata: Metadados adicionais
            
        Returns:
            Caminho do diretório da sessão
        """
        # Criar diretório da sessão
        session_dir = os.path.join(self.base_path, session_name)
        if not os.path.exists(session_dir):
            os.makedirs(session_dir)
        
        # Salvar agente
        agent_filename = f"agent_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        if isinstance(agent, QLearningAgent):
            agent_path = self.save_qlearning_agent(
                agent, os.path.join(session_name, agent_filename), metadata
            )
        elif isinstance(agent, DQNAgent):
            agent_path = self.save_dqn_agent(
                agent, os.path.join(session_name, agent_filename), metadata
            )
        else:
            raise ValueError(f"Tipo de agente não suportado: {type(agent)}")
        
        # Salvar histórico
        history_data = {
            "type": "training_history",
            "timestamp": datetime.now().isoformat(),
            "history": training_history,
            "metadata": metadata or {}
        }
        
        history_path = os.path.join(session_dir, "training_history.json")
        with open(history_path, 'w', encoding='utf-8') as f:
            json.dump(history_data, f, indent=2, ensure_ascii=False)
        
        return session_dir
    
    def list_saved_models(self) -> Dict[str, list]:
        """
        Lista todos os modelos salvos
        
        Returns:
            Dicionário com listas de modelos por tipo
        """
        models = {
            "neural_networks": [],
            "qlearning_agents": [],
            "dqn_agents": [],
            "training_sessions": []
        }
        
        if not os.path.exists(self.base_path):
            return models
        
        for item in os.listdir(self.base_path):
            item_path = os.path.join(self.base_path, item)
            
            if os.path.isfile(item_path) and item.endswith('.json'):
                try:
                    with open(item_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    model_type = data.get('type', 'unknown')
                    if model_type == 'neural_network':
                        models["neural_networks"].append({
                            "filename": item,
                            "timestamp": data.get('timestamp'),
                            "metadata": data.get('metadata', {})
                        })
                    elif model_type == 'qlearning_agent':
                        models["qlearning_agents"].append({
                            "filename": item,
                            "timestamp": data.get('timestamp'),
                            "metadata": data.get('metadata', {})
                        })
                    elif model_type == 'dqn_agent':
                        models["dqn_agents"].append({
                            "filename": item,
                            "timestamp": data.get('timestamp'),
                            "metadata": data.get('metadata', {})
                        })
                except:
                    continue
            
            elif os.path.isdir(item_path):
                models["training_sessions"].append({
                    "session_name": item,
                    "path": item_path
                })
        
        return models
    
    def get_model_info(self, filepath: str) -> Dict[str, Any]:
        """
        Obtém informações sobre um modelo salvo
        
        Args:
            filepath: Caminho para o arquivo do modelo
            
        Returns:
            Informações do modelo
        """
        full_path = os.path.join(self.base_path, filepath) if not os.path.isabs(filepath) else filepath
        
        with open(full_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        info = {
            "type": data.get('type'),
            "timestamp": data.get('timestamp'),
            "metadata": data.get('metadata', {}),
            "file_size": os.path.getsize(full_path)
        }
        
        # Adicionar informações específicas por tipo
        if data.get('type') == 'neural_network':
            network_data = data.get('network', {})
            info['architecture'] = network_data.get('layer_sizes', [])
            info['activations'] = network_data.get('activations', [])
            info['total_parameters'] = self._count_parameters(network_data)
        
        elif data.get('type') in ['qlearning_agent', 'dqn_agent']:
            agent_data = data.get('agent', {})
            info['state_size'] = agent_data.get('state_size')
            info['action_size'] = agent_data.get('action_size')
            info['learning_rate'] = agent_data.get('learning_rate')
            info['step_count'] = agent_data.get('step_count')
            info['policy_type'] = agent_data.get('policy_type')
        
        return info
    
    def _count_parameters(self, network_data: Dict[str, Any]) -> int:
        """Conta o número total de parâmetros em uma rede"""
        total = 0
        layers_data = network_data.get('layers', [])
        
        for layer_data in layers_data:
            neurons_data = layer_data.get('neurons', [])
            for neuron_data in neurons_data:
                weights = neuron_data.get('weights', [])
                total += len(weights) + 1  # +1 para o bias
        
        return total


# Função de conveniência
def save_model(model, filename: str, model_type: str = "auto", 
               metadata: Dict[str, Any] = None, base_path: str = "models") -> str:
    """
    Função de conveniência para salvar modelos
    
    Args:
        model: Modelo a ser salvo
        filename: Nome do arquivo
        model_type: Tipo do modelo ("auto", "neural_network", "qlearning", "dqn")
        metadata: Metadados adicionais
        base_path: Diretório base
        
    Returns:
        Caminho do arquivo salvo
    """
    saver = ModelSaver(base_path)
    
    if model_type == "auto":
        if isinstance(model, NeuralNetwork):
            return saver.save_neural_network(model, filename, metadata)
        elif isinstance(model, QLearningAgent):
            return saver.save_qlearning_agent(model, filename, metadata)
        elif isinstance(model, DQNAgent):
            return saver.save_dqn_agent(model, filename, metadata)
        else:
            raise ValueError(f"Tipo de modelo não reconhecido: {type(model)}")
    
    elif model_type == "neural_network":
        return saver.save_neural_network(model, filename, metadata)
    elif model_type == "qlearning":
        return saver.save_qlearning_agent(model, filename, metadata)
    elif model_type == "dqn":
        return saver.save_dqn_agent(model, filename, metadata)
    else:
        raise ValueError(f"Tipo de modelo não suportado: {model_type}")
