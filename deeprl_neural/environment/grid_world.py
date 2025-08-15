#!/usr/bin/env python3
"""
Grid World Environment - Ambiente de mundo em grade para reinforcement learning
Implementação em Python puro de um ambiente de mundo em grade simples
"""

import random
from typing import List, Tuple, Dict, Any, Optional
from enum import Enum


class GridWorldAction(Enum):
    """Ações possíveis no Grid World"""
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3


class GridWorldCell(Enum):
    """Tipos de células no Grid World"""
    EMPTY = 0
    WALL = 1
    START = 2
    GOAL = 3
    PENALTY = 4


class GridWorld:
    """
    Ambiente de mundo em grade para reinforcement learning.
    
    O agente deve navegar do ponto inicial até o objetivo,
    evitando paredes e células de penalidade.
    """
    
    def __init__(self, 
                 grid_size: int = 3,
                 start_pos: Optional[Tuple[int, int]] = None,
                 goal_pos: Optional[Tuple[int, int]] = None,
                 walls: Optional[List[Tuple[int, int]]] = None,
                 penalties: Optional[List[Tuple[int, int]]] = None,
                 step_penalty: float = -0.1,
                 goal_reward: float = 10.0,
                 penalty_reward: float = -5.0,
                 wall_penalty: float = -1.0,
                 max_steps: int = 100):
        """
        Inicializa o Grid World
        
        Args:
            grid_size: Tamanho da grade (NxN)
            start_pos: Posição inicial (linha, coluna)
            goal_pos: Posição do objetivo
            walls: Lista de posições com paredes
            penalties: Lista de posições com penalidades
            step_penalty: Penalidade por cada passo
            goal_reward: Recompensa por alcançar o objetivo
            penalty_reward: Penalidade por entrar em célula de penalidade
            wall_penalty: Penalidade por tentar atravessar parede
            max_steps: Número máximo de passos por episódio
        """
        self.grid_size = grid_size
        self.start_pos = start_pos or (0, 0)
        self.goal_pos = goal_pos or (grid_size - 1, grid_size - 1)
        self.walls = walls or []
        self.penalties = penalties or []
        
        # Recompensas
        self.step_penalty = step_penalty
        self.goal_reward = goal_reward
        self.penalty_reward = penalty_reward
        self.wall_penalty = wall_penalty
        
        # Estado do jogo
        self.current_pos = self.start_pos
        self.step_count = 0
        self.max_steps = max_steps
        self.done = False
        
        # Criar grid visual
        self.grid = self._create_grid()
        
        # Estatísticas
        self.episode_count = 0
        self.total_reward = 0.0
        
    def _create_grid(self) -> List[List[GridWorldCell]]:
        """Cria a representação visual da grade"""
        grid = [[GridWorldCell.EMPTY for _ in range(self.grid_size)] 
                for _ in range(self.grid_size)]
        
        # Marcar posições especiais
        grid[self.start_pos[0]][self.start_pos[1]] = GridWorldCell.START
        grid[self.goal_pos[0]][self.goal_pos[1]] = GridWorldCell.GOAL
        
        for wall_pos in self.walls:
            if self._is_valid_position(wall_pos):
                grid[wall_pos[0]][wall_pos[1]] = GridWorldCell.WALL
        
        for penalty_pos in self.penalties:
            if self._is_valid_position(penalty_pos):
                grid[penalty_pos[0]][penalty_pos[1]] = GridWorldCell.PENALTY
        
        return grid
    
    def _is_valid_position(self, pos: Tuple[int, int]) -> bool:
        """Verifica se uma posição é válida na grade"""
        row, col = pos
        return 0 <= row < self.grid_size and 0 <= col < self.grid_size
    
    def _is_wall(self, pos: Tuple[int, int]) -> bool:
        """Verifica se uma posição é uma parede"""
        return pos in self.walls
    
    def _get_next_position(self, action: GridWorldAction) -> Tuple[int, int]:
        """Calcula a próxima posição baseada na ação"""
        row, col = self.current_pos
        
        if action == GridWorldAction.UP:
            return (row - 1, col)
        elif action == GridWorldAction.DOWN:
            return (row + 1, col)
        elif action == GridWorldAction.LEFT:
            return (row, col - 1)
        elif action == GridWorldAction.RIGHT:
            return (row, col + 1)
        else:
            return self.current_pos
    
    def step(self, action: int) -> Tuple[List[int], float, bool, Dict[str, Any]]:
        """
        Executa uma ação no ambiente
        
        Args:
            action: Ação a ser executada (0=UP, 1=DOWN, 2=LEFT, 3=RIGHT)
            
        Returns:
            Tupla com (próximo_estado, recompensa, done, info)
        """
        if self.done:
            return self.get_state(), 0.0, True, {"error": "Episode already finished"}
        
        # Converter ação para enum
        try:
            grid_action = GridWorldAction(action)
        except ValueError:
            return self.get_state(), self.wall_penalty, False, {"error": "Invalid action"}
        
        # Calcular próxima posição
        next_pos = self._get_next_position(grid_action)
        
        # Verificar se movimento é válido
        reward = self.step_penalty  # Penalidade básica por passo
        info = {"action": action, "from": self.current_pos, "to": next_pos}
        
        if not self._is_valid_position(next_pos):
            # Tentou sair da grade
            reward = self.wall_penalty
            info["collision"] = "boundary"
        elif self._is_wall(next_pos):
            # Tentou atravessar parede
            reward = self.wall_penalty
            info["collision"] = "wall"
        else:
            # Movimento válido
            self.current_pos = next_pos
            info["moved"] = True
            
            # Verificar tipo de célula
            if next_pos == self.goal_pos:
                reward = self.goal_reward
                self.done = True
                info["goal_reached"] = True
            elif next_pos in self.penalties:
                reward = self.penalty_reward
                info["penalty"] = True
        
        # Incrementar contador de passos
        self.step_count += 1
        if self.step_count >= self.max_steps:
            self.done = True
            info["max_steps_reached"] = True
        
        # Atualizar recompensa total
        self.total_reward += reward
        
        return self.get_state(), reward, self.done, info
    
    def reset(self) -> List[int]:
        """
        Reseta o ambiente para o estado inicial
        
        Returns:
            Estado inicial
        """
        self.current_pos = self.start_pos
        self.step_count = 0
        self.done = False
        self.episode_count += 1
        self.total_reward = 0.0
        
        return self.get_state()
    
    def get_state(self) -> List[int]:
        """
        Obtém o estado atual do ambiente
        
        Returns:
            Estado como lista de inteiros [linha, coluna]
        """
        return [self.current_pos[0], self.current_pos[1]]
    
    def get_state_size(self) -> int:
        """Retorna o tamanho do espaço de estados"""
        return 2  # (linha, coluna)
    
    def get_action_size(self) -> int:
        """Retorna o número de ações possíveis"""
        return len(GridWorldAction)
    
    def get_state_index(self, state: Optional[List[int]] = None) -> int:
        """
        Converte estado (linha, coluna) para índice único
        
        Args:
            state: Estado a converter (usa estado atual se None)
            
        Returns:
            Índice único do estado
        """
        if state is None:
            state = self.get_state()
        
        row, col = state
        return row * self.grid_size + col
    
    def get_state_from_index(self, index: int) -> List[int]:
        """
        Converte índice único para estado (linha, coluna)
        
        Args:
            index: Índice único
            
        Returns:
            Estado como [linha, coluna]
        """
        row = index // self.grid_size
        col = index % self.grid_size
        return [row, col]
    
    def get_total_states(self) -> int:
        """Retorna o número total de estados possíveis"""
        return self.grid_size * self.grid_size
    
    def render(self, mode: str = 'text') -> str:
        """
        Renderiza o estado atual do ambiente
        
        Args:
            mode: Modo de renderização ('text' ou 'symbols')
            
        Returns:
            Representação visual do ambiente
        """
        if mode == 'text':
            return self._render_text()
        elif mode == 'symbols':
            return self._render_symbols()
        else:
            return self._render_text()
    
    def _render_text(self) -> str:
        """Renderiza usando texto descritivo"""
        output = []
        output.append(f"Grid World {self.grid_size}x{self.grid_size}")
        output.append(f"Episode: {self.episode_count}, Step: {self.step_count}/{self.max_steps}")
        output.append(f"Position: {self.current_pos}, Total Reward: {self.total_reward:.2f}")
        output.append(f"Goal: {self.goal_pos}, Done: {self.done}")
        output.append("")
        
        # Grid visual
        for i in range(self.grid_size):
            row_str = ""
            for j in range(self.grid_size):
                if (i, j) == self.current_pos:
                    row_str += "A "  # Agent
                elif (i, j) == self.goal_pos:
                    row_str += "G "  # Goal
                elif (i, j) in self.walls:
                    row_str += "# "  # Wall
                elif (i, j) in self.penalties:
                    row_str += "X "  # Penalty
                elif (i, j) == self.start_pos:
                    row_str += "S "  # Start
                else:
                    row_str += ". "  # Empty
            output.append(row_str)
        
        return "\n".join(output)
    
    def _render_symbols(self) -> str:
        """Renderiza usando símbolos compactos"""
        output = []
        for i in range(self.grid_size):
            row_str = ""
            for j in range(self.grid_size):
                if (i, j) == self.current_pos:
                    row_str += "A"
                elif (i, j) == self.goal_pos:
                    row_str += "G"
                elif (i, j) in self.walls:
                    row_str += "#"
                elif (i, j) in self.penalties:
                    row_str += "X"
                elif (i, j) == self.start_pos:
                    row_str += "S"
                else:
                    row_str += "."
            output.append(row_str)
        
        return "\n".join(output)
    
    def get_info(self) -> Dict[str, Any]:
        """
        Obtém informações completas do ambiente
        
        Returns:
            Dicionário com informações do ambiente
        """
        return {
            "grid_size": self.grid_size,
            "start_pos": self.start_pos,
            "goal_pos": self.goal_pos,
            "current_pos": self.current_pos,
            "walls": self.walls,
            "penalties": self.penalties,
            "step_count": self.step_count,
            "max_steps": self.max_steps,
            "episode_count": self.episode_count,
            "total_reward": self.total_reward,
            "done": self.done,
            "state_size": self.get_state_size(),
            "action_size": self.get_action_size(),
            "total_states": self.get_total_states()
        }
    
    def is_terminal_state(self, state: Optional[List[int]] = None) -> bool:
        """
        Verifica se um estado é terminal
        
        Args:
            state: Estado a verificar (usa estado atual se None)
            
        Returns:
            True se o estado é terminal
        """
        if state is None:
            state = self.get_state()
        
        pos = tuple(state)
        return pos == self.goal_pos
    
    def get_valid_actions(self, state: Optional[List[int]] = None) -> List[int]:
        """
        Obtém ações válidas para um estado
        
        Args:
            state: Estado a verificar (usa estado atual se None)
            
        Returns:
            Lista de ações válidas
        """
        if state is None:
            state = self.get_state()
        
        valid_actions = []
        current_pos = tuple(state)
        
        for action_int in range(self.get_action_size()):
            action = GridWorldAction(action_int)
            
            # Simular movimento
            if action == GridWorldAction.UP:
                next_pos = (current_pos[0] - 1, current_pos[1])
            elif action == GridWorldAction.DOWN:
                next_pos = (current_pos[0] + 1, current_pos[1])
            elif action == GridWorldAction.LEFT:
                next_pos = (current_pos[0], current_pos[1] - 1)
            elif action == GridWorldAction.RIGHT:
                next_pos = (current_pos[0], current_pos[1] + 1)
            else:
                continue
            
            # Verificar se é válido
            if (self._is_valid_position(next_pos) and 
                not self._is_wall(next_pos)):
                valid_actions.append(action_int)
        
        return valid_actions if valid_actions else [0, 1, 2, 3]  # Todas se nenhuma válida
    
    def sample_random_action(self) -> int:
        """Amostra uma ação aleatória válida"""
        valid_actions = self.get_valid_actions()
        return random.choice(valid_actions)
    
    def copy(self) -> 'GridWorld':
        """Cria uma cópia do ambiente"""
        return GridWorld(
            grid_size=self.grid_size,
            start_pos=self.start_pos,
            goal_pos=self.goal_pos,
            walls=self.walls.copy(),
            penalties=self.penalties.copy(),
            step_penalty=self.step_penalty,
            goal_reward=self.goal_reward,
            penalty_reward=self.penalty_reward,
            wall_penalty=self.wall_penalty,
            max_steps=self.max_steps
        )


# Funções utilitárias para criar ambientes predefinidos
def create_simple_grid(size: int = 3) -> GridWorld:
    """
    Cria um Grid World simples sem obstáculos
    
    Args:
        size: Tamanho da grade
        
    Returns:
        Grid World simples
    """
    return GridWorld(
        grid_size=size,
        start_pos=(0, 0),
        goal_pos=(size - 1, size - 1)
    )


def create_maze_grid(size: int = 5) -> GridWorld:
    """
    Cria um Grid World com formato de labirinto
    
    Args:
        size: Tamanho da grade
        
    Returns:
        Grid World com obstáculos
    """
    walls = []
    penalties = []
    
    if size >= 5:
        # Criar algumas paredes para formar um labirinto simples
        walls = [(1, 1), (1, 2), (2, 2), (3, 1)]
        penalties = [(1, 3), (3, 3)]
    
    return GridWorld(
        grid_size=size,
        start_pos=(0, 0),
        goal_pos=(size - 1, size - 1),
        walls=walls,
        penalties=penalties
    )


def create_random_grid(size: int = 4, 
                      wall_density: float = 0.2,
                      penalty_density: float = 0.1) -> GridWorld:
    """
    Cria um Grid World com obstáculos aleatórios
    
    Args:
        size: Tamanho da grade
        wall_density: Densidade de paredes (0.0 a 1.0)
        penalty_density: Densidade de penalidades (0.0 a 1.0)
        
    Returns:
        Grid World com obstáculos aleatórios
    """
    start_pos = (0, 0)
    goal_pos = (size - 1, size - 1)
    
    walls = []
    penalties = []
    
    for i in range(size):
        for j in range(size):
            pos = (i, j)
            
            # Não colocar obstáculos no início e objetivo
            if pos in [start_pos, goal_pos]:
                continue
            
            if random.random() < wall_density:
                walls.append(pos)
            elif random.random() < penalty_density:
                penalties.append(pos)
    
    return GridWorld(
        grid_size=size,
        start_pos=start_pos,
        goal_pos=goal_pos,
        walls=walls,
        penalties=penalties
    )
