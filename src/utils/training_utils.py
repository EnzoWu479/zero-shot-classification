#!/usr/bin/env python3
"""
Training Utils - Utilitários para treinamento de modelos
Implementação em Python puro de funções auxiliares para treinamento
"""

import math
import random
from typing import List, Dict, Any, Tuple, Optional, Callable
from collections import deque


class TrainingLogger:
    """
    Logger para acompanhar o progresso do treinamento
    """
    
    def __init__(self, log_interval: int = 10):
        """
        Inicializa o logger
        
        Args:
            log_interval: Intervalo para salvar logs (episódios)
        """
        self.log_interval = log_interval
        self.history = {
            'episode': [],
            'reward': [],
            'loss': [],
            'epsilon': [],
            'steps': [],
            'success': []
        }
        self.current_episode = 0
        self.start_time = None
        
    def log_episode(self, reward: float, loss: float = None, 
                   epsilon: float = None, steps: int = None, 
                   success: bool = False):
        """
        Registra dados de um episódio
        
        Args:
            reward: Recompensa total do episódio
            loss: Loss médio do episódio
            epsilon: Valor atual de epsilon
            steps: Número de passos no episódio
            success: Se o episódio foi bem-sucedido
        """
        self.current_episode += 1
        
        if self.current_episode % self.log_interval == 0:
            self.history['episode'].append(self.current_episode)
            self.history['reward'].append(reward)
            self.history['loss'].append(loss)
            self.history['epsilon'].append(epsilon)
            self.history['steps'].append(steps)
            self.history['success'].append(success)
    
    def get_recent_performance(self, window: int = 100) -> Dict[str, float]:
        """
        Obtém estatísticas de performance recente
        
        Args:
            window: Tamanho da janela para cálculo
            
        Returns:
            Estatísticas de performance
        """
        if not self.history['reward']:
            return {}
        
        recent_rewards = self.history['reward'][-window:]
        recent_success = self.history['success'][-window:]
        
        stats = {
            'avg_reward': sum(recent_rewards) / len(recent_rewards),
            'max_reward': max(recent_rewards),
            'min_reward': min(recent_rewards),
            'success_rate': sum(recent_success) / len(recent_success) if recent_success else 0.0,
            'episodes_logged': len(recent_rewards)
        }
        
        return stats
    
    def print_progress(self, window: int = 100):
        """
        Imprime progresso atual
        
        Args:
            window: Janela para estatísticas
        """
        stats = self.get_recent_performance(window)
        
        if stats:
            print(f"Episode {self.current_episode}")
            print(f"  Avg Reward (last {window}): {stats['avg_reward']:.3f}")
            print(f"  Success Rate: {stats['success_rate']:.1%}")
            print(f"  Max/Min Reward: {stats['max_reward']:.2f}/{stats['min_reward']:.2f}")
    
    def save_to_dict(self) -> Dict[str, Any]:
        """Converte histórico para dicionário"""
        return {
            'history': self.history.copy(),
            'current_episode': self.current_episode,
            'log_interval': self.log_interval
        }


class EarlyStopping:
    """
    Early stopping para evitar overfitting
    """
    
    def __init__(self, patience: int = 20, min_delta: float = 0.001, 
                 mode: str = 'max'):
        """
        Inicializa early stopping
        
        Args:
            patience: Número de episódios sem melhoria para parar
            min_delta: Melhoria mínima considerada significativa
            mode: 'max' para maximizar, 'min' para minimizar
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_score = None
        self.counter = 0
        self.should_stop = False
        
    def check(self, score: float) -> bool:
        """
        Verifica se deve parar o treinamento
        
        Args:
            score: Score atual (recompensa, loss, etc.)
            
        Returns:
            True se deve parar
        """
        if self.best_score is None:
            self.best_score = score
            return False
        
        improved = False
        if self.mode == 'max':
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
        
        if self.counter >= self.patience:
            self.should_stop = True
        
        return self.should_stop


class LearningRateScheduler:
    """
    Agendador de taxa de aprendizado
    """
    
    def __init__(self, initial_lr: float = 0.01, schedule_type: str = 'constant'):
        """
        Inicializa o agendador
        
        Args:
            initial_lr: Taxa de aprendizado inicial
            schedule_type: Tipo de agenda ('constant', 'linear', 'exponential', 'step')
        """
        self.initial_lr = initial_lr
        self.current_lr = initial_lr
        self.schedule_type = schedule_type
        self.step_count = 0
        
    def get_lr(self, episode: int = None, total_episodes: int = None) -> float:
        """
        Obtém a taxa de aprendizado atual
        
        Args:
            episode: Episódio atual
            total_episodes: Total de episódios planejados
            
        Returns:
            Taxa de aprendizado atual
        """
        if self.schedule_type == 'constant':
            return self.initial_lr
        
        elif self.schedule_type == 'linear' and episode is not None and total_episodes:
            # Decaimento linear
            progress = episode / total_episodes
            self.current_lr = self.initial_lr * (1 - progress)
            
        elif self.schedule_type == 'exponential':
            # Decaimento exponencial
            decay_rate = 0.995
            self.current_lr = self.initial_lr * (decay_rate ** self.step_count)
            
        elif self.schedule_type == 'step':
            # Decaimento em degraus
            if episode and episode % 100 == 0:
                self.current_lr *= 0.9
        
        self.step_count += 1
        return max(self.current_lr, 0.0001)  # LR mínimo


class RewardShaper:
    """
    Moldador de recompensas para melhorar aprendizado
    """
    
    def __init__(self, shaping_type: str = 'none'):
        """
        Inicializa o moldador
        
        Args:
            shaping_type: Tipo de moldagem ('none', 'distance', 'potential')
        """
        self.shaping_type = shaping_type
        self.last_potential = 0.0
        
    def shape_reward(self, state: List[int], next_state: List[int], 
                    original_reward: float, goal_state: List[int]) -> float:
        """
        Molda a recompensa baseada no estado
        
        Args:
            state: Estado atual
            next_state: Próximo estado
            original_reward: Recompensa original
            goal_state: Estado objetivo
            
        Returns:
            Recompensa moldada
        """
        if self.shaping_type == 'none':
            return original_reward
        
        elif self.shaping_type == 'distance':
            # Recompensa baseada na distância ao objetivo
            current_dist = self._manhattan_distance(state, goal_state)
            next_dist = self._manhattan_distance(next_state, goal_state)
            distance_reward = (current_dist - next_dist) * 0.1
            return original_reward + distance_reward
        
        elif self.shaping_type == 'potential':
            # Potential-based reward shaping
            current_potential = -self._manhattan_distance(state, goal_state)
            next_potential = -self._manhattan_distance(next_state, goal_state)
            potential_reward = next_potential - current_potential
            return original_reward + potential_reward
        
        return original_reward
    
    def _manhattan_distance(self, pos1: List[int], pos2: List[int]) -> float:
        """Calcula distância de Manhattan entre duas posições"""
        return sum(abs(a - b) for a, b in zip(pos1, pos2))


class ExperienceBuffer:
    """
    Buffer circular para armazenar experiências
    """
    
    def __init__(self, capacity: int = 1000):
        """
        Inicializa o buffer
        
        Args:
            capacity: Capacidade máxima do buffer
        """
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        
    def add(self, experience: Dict[str, Any]):
        """Adiciona uma experiência ao buffer"""
        self.buffer.append(experience)
        
    def sample(self, batch_size: int) -> List[Dict[str, Any]]:
        """Amostra experiências aleatórias"""
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)
        
        return random.sample(list(self.buffer), batch_size)
    
    def get_recent(self, n: int) -> List[Dict[str, Any]]:
        """Obtém as n experiências mais recentes"""
        return list(self.buffer)[-n:]
    
    def clear(self):
        """Limpa o buffer"""
        self.buffer.clear()
    
    def size(self) -> int:
        """Retorna o tamanho atual do buffer"""
        return len(self.buffer)


def calculate_moving_average(values: List[float], window: int = 10) -> List[float]:
    """
    Calcula média móvel de uma lista de valores
    
    Args:
        values: Lista de valores
        window: Tamanho da janela
        
    Returns:
        Lista com médias móveis
    """
    if len(values) < window:
        return values
    
    moving_avg = []
    for i in range(len(values)):
        if i < window - 1:
            moving_avg.append(values[i])
        else:
            avg = sum(values[i - window + 1:i + 1]) / window
            moving_avg.append(avg)
    
    return moving_avg


def normalize_rewards(rewards: List[float]) -> List[float]:
    """
    Normaliza uma lista de recompensas
    
    Args:
        rewards: Lista de recompensas
        
    Returns:
        Lista normalizada
    """
    if not rewards:
        return rewards
    
    mean_reward = sum(rewards) / len(rewards)
    variance = sum((r - mean_reward) ** 2 for r in rewards) / len(rewards)
    std_reward = math.sqrt(variance) if variance > 0 else 1.0
    
    return [(r - mean_reward) / std_reward for r in rewards]


def calculate_success_rate(results: List[bool], window: int = 100) -> float:
    """
    Calcula taxa de sucesso em uma janela
    
    Args:
        results: Lista de resultados booleanos
        window: Tamanho da janela
        
    Returns:
        Taxa de sucesso (0.0 a 1.0)
    """
    if not results:
        return 0.0
    
    recent_results = results[-window:] if len(results) > window else results
    return sum(recent_results) / len(recent_results)


def epsilon_schedule(episode: int, max_episodes: int, 
                    epsilon_start: float = 1.0, epsilon_end: float = 0.01,
                    schedule_type: str = 'linear') -> float:
    """
    Calcula epsilon para uma agenda específica
    
    Args:
        episode: Episódio atual
        max_episodes: Total de episódios
        epsilon_start: Valor inicial de epsilon
        epsilon_end: Valor final de epsilon
        schedule_type: Tipo de agenda ('linear', 'exponential', 'cosine')
        
    Returns:
        Valor de epsilon
    """
    if episode >= max_episodes:
        return epsilon_end
    
    progress = episode / max_episodes
    
    if schedule_type == 'linear':
        return epsilon_start - (epsilon_start - epsilon_end) * progress
    
    elif schedule_type == 'exponential':
        decay_rate = math.log(epsilon_end / epsilon_start) / max_episodes
        return epsilon_start * math.exp(decay_rate * episode)
    
    elif schedule_type == 'cosine':
        return epsilon_end + (epsilon_start - epsilon_end) * 0.5 * (1 + math.cos(math.pi * progress))
    
    return epsilon_start


def create_training_curriculum(difficulty_levels: List[Dict[str, Any]], 
                              episodes_per_level: int = 100) -> List[Dict[str, Any]]:
    """
    Cria um currículo de treinamento progressivo
    
    Args:
        difficulty_levels: Lista de configurações de dificuldade
        episodes_per_level: Episódios por nível
        
    Returns:
        Lista de configurações para cada episódio
    """
    curriculum = []
    
    for level in difficulty_levels:
        for _ in range(episodes_per_level):
            curriculum.append(level.copy())
    
    return curriculum


def save_training_config(config: Dict[str, Any], filepath: str = "training_config.json"):
    """
    Salva configuração de treinamento
    
    Args:
        config: Dicionário de configuração
        filepath: Caminho do arquivo
    """
    import json
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)


def load_training_config(filepath: str = "training_config.json") -> Dict[str, Any]:
    """
    Carrega configuração de treinamento
    
    Args:
        filepath: Caminho do arquivo
        
    Returns:
        Dicionário de configuração
    """
    import json
    
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


class PerformanceMetrics:
    """
    Calculador de métricas de performance
    """
    
    def __init__(self):
        self.metrics = {}
        
    def calculate_all_metrics(self, rewards: List[float], 
                             success_flags: List[bool],
                             episode_lengths: List[int]) -> Dict[str, float]:
        """
        Calcula todas as métricas de performance
        
        Args:
            rewards: Lista de recompensas por episódio
            success_flags: Lista de flags de sucesso
            episode_lengths: Lista de comprimentos de episódio
            
        Returns:
            Dicionário com métricas
        """
        metrics = {}
        
        if rewards:
            metrics['avg_reward'] = sum(rewards) / len(rewards)
            metrics['max_reward'] = max(rewards)
            metrics['min_reward'] = min(rewards)
            metrics['std_reward'] = self._calculate_std(rewards)
            
        if success_flags:
            metrics['success_rate'] = sum(success_flags) / len(success_flags)
            
        if episode_lengths:
            metrics['avg_episode_length'] = sum(episode_lengths) / len(episode_lengths)
            metrics['min_episode_length'] = min(episode_lengths)
            
        # Métricas de convergência
        if len(rewards) > 10:
            metrics['convergence_stability'] = self._calculate_convergence_stability(rewards)
            
        return metrics
    
    def _calculate_std(self, values: List[float]) -> float:
        """Calcula desvio padrão"""
        if len(values) < 2:
            return 0.0
            
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        return math.sqrt(variance)
    
    def _calculate_convergence_stability(self, rewards: List[float], 
                                       window: int = 50) -> float:
        """
        Calcula estabilidade de convergência
        
        Args:
            rewards: Lista de recompensas
            window: Tamanho da janela para análise
            
        Returns:
            Score de estabilidade (0.0 a 1.0, maior é melhor)
        """
        if len(rewards) < window * 2:
            return 0.0
        
        # Comparar primeiras e últimas janelas
        first_window = rewards[:window]
        last_window = rewards[-window:]
        
        first_avg = sum(first_window) / len(first_window)
        last_avg = sum(last_window) / len(last_window)
        
        # Se melhorou, calcular estabilidade da última janela
        if last_avg > first_avg:
            last_std = self._calculate_std(last_window)
            stability = 1.0 / (1.0 + last_std)  # Menor variação = maior estabilidade
            return min(stability, 1.0)
        
        return 0.0
