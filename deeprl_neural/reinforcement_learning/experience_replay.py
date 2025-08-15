"""
Experience Replay Buffer para algoritmos de Deep Reinforcement Learning.
Implementação pura em Python sem dependências externas.
"""

import random
from typing import List, Tuple, Union, Optional, Any
from collections import deque


class Experience:
    """
    Representa uma experiência individual no buffer.
    
    Uma experiência contém:
    - Estado atual (state)
    - Ação tomada (action)
    - Recompensa recebida (reward)
    - Próximo estado (next_state)
    - Flag de terminal (done)
    """
    
    def __init__(self, 
                 state: Union[List[float], int],
                 action: Union[int, List[float]],
                 reward: float,
                 next_state: Union[List[float], int],
                 done: bool):
        """
        Inicializa uma experiência.
        
        Args:
            state: Estado atual
            action: Ação tomada
            reward: Recompensa recebida
            next_state: Próximo estado
            done: True se o episódio terminou
        """
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.done = done
    
    def to_tuple(self) -> Tuple[Any, Any, float, Any, bool]:
        """Converte para tupla para compatibilidade"""
        return (self.state, self.action, self.reward, self.next_state, self.done)
    
    def __repr__(self) -> str:
        return f"Experience(state={self.state}, action={self.action}, reward={self.reward:.3f}, done={self.done})"


class ExperienceReplayBuffer:
    """
    Buffer de experiência para armazenar e amostrar experiências de treinamento.
    
    Implementa:
    - Armazenamento circular (FIFO quando cheio)
    - Amostragem aleatória uniforme
    - Diferentes estratégias de priorização (futuro)
    """
    
    def __init__(self, 
                 max_size: int = 10000,
                 seed: Optional[int] = None):
        """
        Inicializa o buffer de experiência.
        
        Args:
            max_size: Tamanho máximo do buffer
            seed: Seed para reprodutibilidade
        """
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
        self.size = 0
        
        if seed is not None:
            random.seed(seed)
    
    def add(self, 
           state: Union[List[float], int],
           action: Union[int, List[float]],
           reward: float,
           next_state: Union[List[float], int],
           done: bool):
        """
        Adiciona uma nova experiência ao buffer.
        
        Args:
            state: Estado atual
            action: Ação tomada
            reward: Recompensa recebida
            next_state: Próximo estado
            done: True se o episódio terminou
        """
        experience = Experience(state, action, reward, next_state, done)
        self.buffer.append(experience)
        self.size = min(self.size + 1, self.max_size)
    
    def add_experience(self, experience: Experience):
        """Adiciona uma experiência já criada ao buffer"""
        self.buffer.append(experience)
        self.size = min(self.size + 1, self.max_size)
    
    def sample(self, batch_size: int) -> List[Experience]:
        """
        Amostra um batch de experiências aleatoriamente.
        
        Args:
            batch_size: Tamanho do batch
            
        Returns:
            Lista de experiências amostradas
        """
        if batch_size > self.size:
            raise ValueError(f"Batch size ({batch_size}) maior que buffer size ({self.size})")
        
        return random.sample(list(self.buffer), batch_size)
    
    def sample_tuples(self, batch_size: int) -> Tuple[List, List, List[float], List, List[bool]]:
        """
        Amostra experiências e retorna como tuplas separadas.
        
        Args:
            batch_size: Tamanho do batch
            
        Returns:
            Tupla com (states, actions, rewards, next_states, dones)
        """
        experiences = self.sample(batch_size)
        
        states = [exp.state for exp in experiences]
        actions = [exp.action for exp in experiences]
        rewards = [exp.reward for exp in experiences]
        next_states = [exp.next_state for exp in experiences]
        dones = [exp.done for exp in experiences]
        
        return states, actions, rewards, next_states, dones
    
    def get_latest(self, n: int = 1) -> List[Experience]:
        """
        Retorna as n experiências mais recentes.
        
        Args:
            n: Número de experiências
            
        Returns:
            Lista das experiências mais recentes
        """
        if n > self.size:
            n = self.size
        
        return list(self.buffer)[-n:]
    
    def get_oldest(self, n: int = 1) -> List[Experience]:
        """
        Retorna as n experiências mais antigas.
        
        Args:
            n: Número de experiências
            
        Returns:
            Lista das experiências mais antigas
        """
        if n > self.size:
            n = self.size
        
        return list(self.buffer)[:n]
    
    def clear(self):
        """Limpa todo o buffer"""
        self.buffer.clear()
        self.size = 0
    
    def is_full(self) -> bool:
        """Verifica se o buffer está cheio"""
        return self.size >= self.max_size
    
    def is_empty(self) -> bool:
        """Verifica se o buffer está vazio"""
        return self.size == 0
    
    def can_sample(self, batch_size: int) -> bool:
        """Verifica se é possível amostrar um batch do tamanho especificado"""
        return self.size >= batch_size
    
    def get_size(self) -> int:
        """Retorna o tamanho atual do buffer"""
        return self.size
    
    def get_max_size(self) -> int:
        """Retorna o tamanho máximo do buffer"""
        return self.max_size
    
    def get_usage_percentage(self) -> float:
        """Retorna a porcentagem de uso do buffer"""
        return (self.size / self.max_size) * 100.0
    
    def get_statistics(self) -> dict:
        """Retorna estatísticas do buffer"""
        if self.is_empty():
            return {
                'size': 0,
                'max_size': self.max_size,
                'usage_percentage': 0.0,
                'avg_reward': 0.0,
                'min_reward': 0.0,
                'max_reward': 0.0,
                'terminal_episodes': 0
            }
        
        rewards = [exp.reward for exp in self.buffer]
        terminal_count = sum(1 for exp in self.buffer if exp.done)
        
        return {
            'size': self.size,
            'max_size': self.max_size,
            'usage_percentage': self.get_usage_percentage(),
            'avg_reward': sum(rewards) / len(rewards),
            'min_reward': min(rewards),
            'max_reward': max(rewards),
            'terminal_episodes': terminal_count
        }
    
    def filter_by_reward(self, min_reward: float = None, max_reward: float = None) -> List[Experience]:
        """
        Filtra experiências por recompensa.
        
        Args:
            min_reward: Recompensa mínima
            max_reward: Recompensa máxima
            
        Returns:
            Lista de experiências filtradas
        """
        filtered = []
        
        for exp in self.buffer:
            if min_reward is not None and exp.reward < min_reward:
                continue
            if max_reward is not None and exp.reward > max_reward:
                continue
            filtered.append(exp)
        
        return filtered
    
    def filter_terminal_experiences(self) -> List[Experience]:
        """Retorna apenas experiências terminais (done=True)"""
        return [exp for exp in self.buffer if exp.done]
    
    def filter_non_terminal_experiences(self) -> List[Experience]:
        """Retorna apenas experiências não-terminais (done=False)"""
        return [exp for exp in self.buffer if not exp.done]
    
    def save_to_file(self, filepath: str):
        """
        Salva o buffer em arquivo (formato simples).
        
        Args:
            filepath: Caminho do arquivo
        """
        import json
        
        data = {
            'max_size': self.max_size,
            'size': self.size,
            'experiences': []
        }
        
        for exp in self.buffer:
            exp_data = {
                'state': exp.state,
                'action': exp.action,
                'reward': exp.reward,
                'next_state': exp.next_state,
                'done': exp.done
            }
            data['experiences'].append(exp_data)
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load_from_file(cls, filepath: str) -> 'ExperienceReplayBuffer':
        """
        Carrega buffer de arquivo.
        
        Args:
            filepath: Caminho do arquivo
            
        Returns:
            Buffer carregado
        """
        import json
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        buffer = cls(max_size=data['max_size'])
        
        for exp_data in data['experiences']:
            buffer.add(
                state=exp_data['state'],
                action=exp_data['action'],
                reward=exp_data['reward'],
                next_state=exp_data['next_state'],
                done=exp_data['done']
            )
        
        return buffer
    
    def __len__(self) -> int:
        """Suporte para len()"""
        return self.size
    
    def __str__(self) -> str:
        """Representação string do buffer"""
        return f"ExperienceReplayBuffer(size={self.size}/{self.max_size}, usage={self.get_usage_percentage():.1f}%)"
    
    def __repr__(self) -> str:
        return f"ExperienceReplayBuffer({self.max_size})"


class PrioritizedExperienceReplayBuffer(ExperienceReplayBuffer):
    """
    Versão com priorização das experiências baseada no erro TD.
    
    Implementação simplificada sem heap para manter compatibilidade
    com Python puro.
    """
    
    def __init__(self, 
                 max_size: int = 10000,
                 alpha: float = 0.6,
                 beta: float = 0.4,
                 beta_increment: float = 0.001,
                 epsilon: float = 1e-6,
                 seed: Optional[int] = None):
        """
        Inicializa buffer com priorização.
        
        Args:
            max_size: Tamanho máximo do buffer
            alpha: Controla o nível de priorização (0 = uniforme, 1 = total)
            beta: Controla correção de importance sampling
            beta_increment: Incremento de beta por amostragem
            epsilon: Pequeno valor para evitar prioridade zero
            seed: Seed para reprodutibilidade
        """
        super().__init__(max_size, seed)
        
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = epsilon
        
        # Lista de prioridades paralela ao buffer
        self.priorities = deque(maxlen=max_size)
        self.max_priority = 1.0
    
    def add(self, 
           state: Union[List[float], int],
           action: Union[int, List[float]],
           reward: float,
           next_state: Union[List[float], int],
           done: bool,
           priority: Optional[float] = None):
        """Adiciona experiência com prioridade"""
        super().add(state, action, reward, next_state, done)
        
        # Usar prioridade máxima para novas experiências
        if priority is None:
            priority = self.max_priority
        
        self.priorities.append(priority)
    
    def update_priorities(self, indices: List[int], priorities: List[float]):
        """
        Atualiza prioridades de experiências específicas.
        
        Args:
            indices: Índices das experiências
            priorities: Novas prioridades
        """
        for idx, priority in zip(indices, priorities):
            if 0 <= idx < self.size:
                priority = abs(priority) + self.epsilon
                self.priorities[idx] = priority
                self.max_priority = max(self.max_priority, priority)
    
    def sample(self, batch_size: int) -> Tuple[List[Experience], List[int], List[float]]:
        """
        Amostra experiências com base nas prioridades.
        
        Args:
            batch_size: Tamanho do batch
            
        Returns:
            Tupla com (experiências, índices, pesos de importance sampling)
        """
        if batch_size > self.size:
            raise ValueError(f"Batch size ({batch_size}) maior que buffer size ({self.size})")
        
        # Calcular probabilidades de amostragem
        priorities_array = list(self.priorities)[:self.size]
        priorities_powered = [p ** self.alpha for p in priorities_array]
        total_priority = sum(priorities_powered)
        
        if total_priority == 0:
            # Fallback para amostragem uniforme
            probabilities = [1.0 / self.size] * self.size
        else:
            probabilities = [p / total_priority for p in priorities_powered]
        
        # Amostrar índices baseado nas probabilidades
        indices = []
        for _ in range(batch_size):
            rand_val = random.random()
            cumsum = 0.0
            for i, prob in enumerate(probabilities):
                cumsum += prob
                if rand_val <= cumsum:
                    indices.append(i)
                    break
            else:
                indices.append(len(probabilities) - 1)
        
        # Calcular pesos de importance sampling
        min_prob = min(probabilities)
        weights = []
        for idx in indices:
            weight = (self.size * probabilities[idx]) ** (-self.beta)
            weights.append(weight)
        
        # Normalizar pesos
        max_weight = max(weights)
        weights = [w / max_weight for w in weights]
        
        # Incrementar beta
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        # Retornar experiências, índices e pesos
        experiences = [list(self.buffer)[idx] for idx in indices]
        
        return experiences, indices, weights
    
    def clear(self):
        """Limpa buffer e prioridades"""
        super().clear()
        self.priorities.clear()
        self.max_priority = 1.0


# Funções utilitárias
def create_simple_buffer(max_size: int = 10000) -> ExperienceReplayBuffer:
    """Cria um buffer simples com configurações padrão"""
    return ExperienceReplayBuffer(max_size=max_size)


def create_prioritized_buffer(max_size: int = 10000, 
                            alpha: float = 0.6, 
                            beta: float = 0.4) -> PrioritizedExperienceReplayBuffer:
    """Cria um buffer priorizado com configurações padrão"""
    return PrioritizedExperienceReplayBuffer(
        max_size=max_size,
        alpha=alpha,
        beta=beta
    )
