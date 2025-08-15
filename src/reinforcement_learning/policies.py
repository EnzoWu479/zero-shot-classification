"""
Políticas de exploração para algoritmos de Reinforcement Learning.
Implementação pura em Python sem dependências externas.
"""

import random
import math
from typing import List, Union, Optional, Dict, Any
from abc import ABC, abstractmethod


class ExplorationPolicy(ABC):
    """Classe abstrata para políticas de exploração"""
    
    @abstractmethod
    def select_action(self, 
                     q_values: List[float], 
                     step: int = 0, 
                     episode: int = 0) -> int:
        """
        Seleciona uma ação baseada nos Q-values e política de exploração.
        
        Args:
            q_values: Lista de Q-values para cada ação
            step: Passo atual
            episode: Episódio atual
            
        Returns:
            Índice da ação selecionada
        """
        pass
    
    @abstractmethod
    def get_exploration_rate(self, step: int = 0, episode: int = 0) -> float:
        """Retorna a taxa de exploração atual"""
        pass
    
    def update(self, step: int = 0, episode: int = 0):
        """Atualiza a política (para políticas que mudam com o tempo)"""
        pass


class EpsilonGreedyPolicy(ExplorationPolicy):
    """
    Política Epsilon-Greedy: escolhe ação aleatória com probabilidade epsilon,
    senão escolhe a ação com maior Q-value.
    """
    
    def __init__(self, 
                 epsilon_start: float = 1.0,
                 epsilon_end: float = 0.01,
                 epsilon_decay: float = 0.995,
                 decay_type: str = 'exponential',
                 decay_steps: int = None):
        """
        Inicializa política epsilon-greedy.
        
        Args:
            epsilon_start: Valor inicial de epsilon
            epsilon_end: Valor final de epsilon
            epsilon_decay: Fator de decaimento
            decay_type: Tipo de decaimento ('exponential', 'linear', 'step')
            decay_steps: Número de passos para decaimento (para linear)
        """
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.decay_type = decay_type
        self.decay_steps = decay_steps or 10000
        
        self.current_epsilon = epsilon_start
    
    def select_action(self, 
                     q_values: List[float], 
                     step: int = 0, 
                     episode: int = 0) -> int:
        """Seleciona ação usando epsilon-greedy"""
        if random.random() < self.current_epsilon:
            # Exploração: ação aleatória
            return random.randint(0, len(q_values) - 1)
        else:
            # Exploração: melhor ação (greedy)
            return q_values.index(max(q_values))
    
    def get_exploration_rate(self, step: int = 0, episode: int = 0) -> float:
        """Retorna epsilon atual"""
        return self.current_epsilon
    
    def update(self, step: int = 0, episode: int = 0):
        """Atualiza epsilon baseado no tipo de decaimento"""
        if self.decay_type == 'exponential':
            self.current_epsilon = max(
                self.epsilon_end,
                self.current_epsilon * self.epsilon_decay
            )
        elif self.decay_type == 'linear':
            decay_rate = (self.epsilon_start - self.epsilon_end) / self.decay_steps
            self.current_epsilon = max(
                self.epsilon_end,
                self.epsilon_start - decay_rate * step
            )
        elif self.decay_type == 'step':
            # Decaimento em degraus
            if step % self.decay_steps == 0 and step > 0:
                self.current_epsilon = max(
                    self.epsilon_end,
                    self.current_epsilon * self.epsilon_decay
                )
    
    def reset(self):
        """Reseta epsilon para o valor inicial"""
        self.current_epsilon = self.epsilon_start


class BoltzmannPolicy(ExplorationPolicy):
    """
    Política Boltzmann (Softmax): probabilidade de seleção baseada
    na distribuição softmax dos Q-values.
    """
    
    def __init__(self, 
                 temperature_start: float = 1.0,
                 temperature_end: float = 0.1,
                 temperature_decay: float = 0.99):
        """
        Inicializa política Boltzmann.
        
        Args:
            temperature_start: Temperatura inicial (maior = mais exploração)
            temperature_end: Temperatura final
            temperature_decay: Fator de decaimento da temperatura
        """
        self.temperature_start = temperature_start
        self.temperature_end = temperature_end
        self.temperature_decay = temperature_decay
        
        self.current_temperature = temperature_start
    
    def select_action(self, 
                     q_values: List[float], 
                     step: int = 0, 
                     episode: int = 0) -> int:
        """Seleciona ação usando distribuição Boltzmann"""
        if self.current_temperature <= 0:
            # Temperatura zero = greedy
            return q_values.index(max(q_values))
        
        # Calcular probabilidades softmax
        scaled_q_values = [q / self.current_temperature for q in q_values]
        
        # Subtrair máximo para estabilidade numérica
        max_q = max(scaled_q_values)
        exp_values = [math.exp(q - max_q) for q in scaled_q_values]
        
        total_exp = sum(exp_values)
        probabilities = [exp_val / total_exp for exp_val in exp_values]
        
        # Amostragem baseada nas probabilidades
        rand_val = random.random()
        cumsum = 0.0
        
        for i, prob in enumerate(probabilities):
            cumsum += prob
            if rand_val <= cumsum:
                return i
        
        # Fallback (não deveria acontecer)
        return len(probabilities) - 1
    
    def get_exploration_rate(self, step: int = 0, episode: int = 0) -> float:
        """Retorna temperatura atual (proxy para exploração)"""
        return self.current_temperature
    
    def update(self, step: int = 0, episode: int = 0):
        """Atualiza temperatura"""
        self.current_temperature = max(
            self.temperature_end,
            self.current_temperature * self.temperature_decay
        )
    
    def reset(self):
        """Reseta temperatura para valor inicial"""
        self.current_temperature = self.temperature_start


class UCBPolicy(ExplorationPolicy):
    """
    Upper Confidence Bound: seleciona ações baseado no limite superior
    de confiança, balanceando exploração e exploração.
    """
    
    def __init__(self, c: float = 2.0):
        """
        Inicializa política UCB.
        
        Args:
            c: Parâmetro de exploração (maior = mais exploração)
        """
        self.c = c
        self.action_counts = None
        self.total_steps = 0
    
    def select_action(self, 
                     q_values: List[float], 
                     step: int = 0, 
                     episode: int = 0) -> int:
        """Seleciona ação usando UCB"""
        num_actions = len(q_values)
        
        # Inicializar contadores se necessário
        if self.action_counts is None:
            self.action_counts = [0] * num_actions
        
        # Se alguma ação nunca foi tentada, seleciona ela
        for i in range(num_actions):
            if self.action_counts[i] == 0:
                self.action_counts[i] += 1
                self.total_steps += 1
                return i
        
        # Calcular UCB para cada ação
        ucb_values = []
        for i in range(num_actions):
            if self.total_steps == 0:
                ucb = float('inf')
            else:
                confidence = self.c * math.sqrt(
                    math.log(self.total_steps) / self.action_counts[i]
                )
                ucb = q_values[i] + confidence
            ucb_values.append(ucb)
        
        # Selecionar ação com maior UCB
        action = ucb_values.index(max(ucb_values))
        self.action_counts[action] += 1
        self.total_steps += 1
        
        return action
    
    def get_exploration_rate(self, step: int = 0, episode: int = 0) -> float:
        """UCB não tem taxa de exploração direta"""
        if self.total_steps == 0:
            return 1.0
        
        # Retorna uma métrica baseada na distribuição das ações
        if self.action_counts is None:
            return 1.0
        
        total_actions = sum(self.action_counts)
        if total_actions == 0:
            return 1.0
        
        # Entropia normalizada como proxy de exploração
        entropy = 0.0
        for count in self.action_counts:
            if count > 0:
                prob = count / total_actions
                entropy -= prob * math.log(prob)
        
        max_entropy = math.log(len(self.action_counts))
        return entropy / max_entropy if max_entropy > 0 else 0.0
    
    def reset(self):
        """Reseta contadores"""
        self.action_counts = None
        self.total_steps = 0


class ThompsonSamplingPolicy(ExplorationPolicy):
    """
    Thompson Sampling: mantém distribuições de probabilidade sobre
    os valores das ações e amostra delas.
    
    Implementação simplificada usando distribuições beta.
    """
    
    def __init__(self, alpha_init: float = 1.0, beta_init: float = 1.0):
        """
        Inicializa Thompson Sampling.
        
        Args:
            alpha_init: Parâmetro alpha inicial para distribuições beta
            beta_init: Parâmetro beta inicial para distribuições beta
        """
        self.alpha_init = alpha_init
        self.beta_init = beta_init
        self.alpha_params = None
        self.beta_params = None
    
    def select_action(self, 
                     q_values: List[float], 
                     step: int = 0, 
                     episode: int = 0) -> int:
        """Seleciona ação usando Thompson Sampling"""
        num_actions = len(q_values)
        
        # Inicializar parâmetros se necessário
        if self.alpha_params is None:
            self.alpha_params = [self.alpha_init] * num_actions
            self.beta_params = [self.beta_init] * num_actions
        
        # Amostrar de cada distribuição beta
        sampled_values = []
        for i in range(num_actions):
            # Implementação simples de amostragem beta usando q_values
            # (em implementação real, usaria biblioteca estatística)
            normalized_q = (q_values[i] + 1) / 2  # Normalizar para [0,1]
            
            # Amostra simulada baseada nos parâmetros
            random_val = random.random()
            sample = (normalized_q * self.alpha_params[i] + 
                     random_val * self.beta_params[i]) / (
                         self.alpha_params[i] + self.beta_params[i])
            
            sampled_values.append(sample)
        
        return sampled_values.index(max(sampled_values))
    
    def update_parameters(self, action: int, reward: float):
        """
        Atualiza parâmetros da distribuição baseado no resultado.
        
        Args:
            action: Ação tomada
            reward: Recompensa recebida
        """
        if self.alpha_params is None:
            return
        
        # Atualizar parâmetros beta baseado na recompensa
        if reward > 0:
            self.alpha_params[action] += reward
        else:
            self.beta_params[action] += abs(reward)
    
    def get_exploration_rate(self, step: int = 0, episode: int = 0) -> float:
        """Thompson Sampling adapta automaticamente"""
        if self.alpha_params is None:
            return 1.0
        
        # Variância média como proxy de exploração
        total_variance = 0.0
        for i in range(len(self.alpha_params)):
            alpha = self.alpha_params[i]
            beta = self.beta_params[i]
            variance = (alpha * beta) / ((alpha + beta) ** 2 * (alpha + beta + 1))
            total_variance += variance
        
        return total_variance / len(self.alpha_params)
    
    def reset(self):
        """Reseta parâmetros"""
        self.alpha_params = None
        self.beta_params = None


class GreedyPolicy(ExplorationPolicy):
    """Política puramente gananciosa (sem exploração)"""
    
    def select_action(self, 
                     q_values: List[float], 
                     step: int = 0, 
                     episode: int = 0) -> int:
        """Sempre seleciona a melhor ação"""
        return q_values.index(max(q_values))
    
    def get_exploration_rate(self, step: int = 0, episode: int = 0) -> float:
        """Política greedy não explora"""
        return 0.0


class RandomPolicy(ExplorationPolicy):
    """Política puramente aleatória"""
    
    def select_action(self, 
                     q_values: List[float], 
                     step: int = 0, 
                     episode: int = 0) -> int:
        """Sempre seleciona ação aleatória"""
        return random.randint(0, len(q_values) - 1)
    
    def get_exploration_rate(self, step: int = 0, episode: int = 0) -> float:
        """Política aleatória sempre explora"""
        return 1.0


# Factory function para criar políticas
def create_exploration_policy(policy_type: str, **kwargs) -> ExplorationPolicy:
    """
    Factory function para criar políticas de exploração.
    
    Args:
        policy_type: Tipo da política ('epsilon_greedy', 'boltzmann', 'ucb', 'thompson', 'greedy', 'random')
        **kwargs: Parâmetros específicos da política
        
    Returns:
        Instância da política de exploração
    """
    policy_type = policy_type.lower()
    
    if policy_type == 'epsilon_greedy':
        return EpsilonGreedyPolicy(**kwargs)
    elif policy_type == 'boltzmann' or policy_type == 'softmax':
        return BoltzmannPolicy(**kwargs)
    elif policy_type == 'ucb':
        return UCBPolicy(**kwargs)
    elif policy_type == 'thompson':
        return ThompsonSamplingPolicy(**kwargs)
    elif policy_type == 'greedy':
        return GreedyPolicy()
    elif policy_type == 'random':
        return RandomPolicy()
    else:
        raise ValueError(f"Política '{policy_type}' não suportada")


# Classe para comparar políticas
class PolicyComparison:
    """Classe utilitária para comparar diferentes políticas"""
    
    def __init__(self):
        self.policies = {}
        self.results = {}
    
    def add_policy(self, name: str, policy: ExplorationPolicy):
        """Adiciona uma política para comparação"""
        self.policies[name] = policy
        self.results[name] = {
            'actions_selected': [],
            'exploration_rates': []
        }
    
    def simulate_step(self, q_values: List[float], step: int = 0):
        """Simula um passo para todas as políticas"""
        for name, policy in self.policies.items():
            action = policy.select_action(q_values, step)
            exploration_rate = policy.get_exploration_rate(step)
            
            self.results[name]['actions_selected'].append(action)
            self.results[name]['exploration_rates'].append(exploration_rate)
            
            policy.update(step)
    
    def get_summary(self) -> Dict[str, Dict[str, Any]]:
        """Retorna resumo dos resultados"""
        summary = {}
        
        for name, results in self.results.items():
            actions = results['actions_selected']
            exploration_rates = results['exploration_rates']
            
            if actions:
                # Calcular diversidade de ações
                unique_actions = len(set(actions))
                total_actions = len(actions)
                action_diversity = unique_actions / total_actions if total_actions > 0 else 0
                
                # Exploration rate médio
                avg_exploration = sum(exploration_rates) / len(exploration_rates) if exploration_rates else 0
                
                summary[name] = {
                    'total_steps': total_actions,
                    'unique_actions': unique_actions,
                    'action_diversity': action_diversity,
                    'avg_exploration_rate': avg_exploration,
                    'final_exploration_rate': exploration_rates[-1] if exploration_rates else 0
                }
            else:
                summary[name] = {
                    'total_steps': 0,
                    'unique_actions': 0,
                    'action_diversity': 0,
                    'avg_exploration_rate': 0,
                    'final_exploration_rate': 0
                }
        
        return summary


class PolicyFactory:
    """Factory para criação de políticas de exploração"""
    
    @staticmethod
    def create_policy(policy_type: str, action_size: int, **kwargs) -> ExplorationPolicy:
        """
        Cria uma política de exploração do tipo especificado
        
        Args:
            policy_type: Tipo da política ('epsilon_greedy', 'boltzmann', 'ucb', etc.)
            action_size: Número de ações possíveis
            **kwargs: Parâmetros específicos da política
        
        Returns:
            Instância da política especificada
        """
        policy_type = policy_type.lower()
        
        if policy_type == 'epsilon_greedy':
            return EpsilonGreedyPolicy(
                epsilon_start=kwargs.get('initial_epsilon', 1.0),
                epsilon_decay=kwargs.get('decay_rate', 0.995),
                epsilon_end=kwargs.get('min_epsilon', 0.01)
            )
        
        elif policy_type == 'boltzmann':
            return BoltzmannPolicy(
                temperature_start=kwargs.get('initial_temperature', 2.0),
                temperature_decay=kwargs.get('decay_rate', 0.99),
                temperature_end=kwargs.get('min_temperature', 0.1)
            )
        
        elif policy_type == 'ucb':
            return UCBPolicy(
                c=kwargs.get('confidence_level', 1.0)
            )
        
        elif policy_type == 'greedy':
            return GreedyPolicy()
        
        elif policy_type == 'random':
            return RandomPolicy()
        
        elif policy_type == 'thompson_sampling':
            return ThompsonSamplingPolicy(
                alpha_init=kwargs.get('alpha_init', 1.0),
                beta_init=kwargs.get('beta_init', 1.0)
            )
        
        else:
            raise ValueError(f"Tipo de política não suportado: {policy_type}")
    
    @staticmethod
    def get_available_policies() -> list:
        """Retorna lista de políticas disponíveis"""
        return [
            'epsilon_greedy',
            'boltzmann', 
            'ucb',
            'greedy',
            'random',
            'thompson_sampling'
        ]
