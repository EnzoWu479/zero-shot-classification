#!/usr/bin/env python3
"""
Testes para o módulo de environment
"""

import unittest
import sys
import os

# Imports do projeto
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from environment.grid_world import (
    GridWorld, GridWorldAction, GridWorldCell,
    create_simple_grid, create_maze_grid, create_random_grid
)


class TestGridWorld(unittest.TestCase):
    """Testes para GridWorld"""
    
    def setUp(self):
        """Setup para cada teste"""
        self.env = GridWorld(grid_size=3)
    
    def test_init(self):
        """Testa inicialização do ambiente"""
        self.assertEqual(self.env.grid_size, 3)
        self.assertEqual(self.env.start_pos, (0, 0))
        self.assertEqual(self.env.goal_pos, (2, 2))
        self.assertEqual(self.env.current_pos, (0, 0))
        self.assertFalse(self.env.done)
        self.assertEqual(self.env.step_count, 0)
    
    def test_get_state(self):
        """Testa obtenção do estado"""
        state = self.env.get_state()
        self.assertEqual(state, [0, 0])
        
        # Mudar posição e testar novamente
        self.env.current_pos = (1, 2)
        state = self.env.get_state()
        self.assertEqual(state, [1, 2])
    
    def test_get_state_size_and_action_size(self):
        """Testa tamanhos de estado e ação"""
        self.assertEqual(self.env.get_state_size(), 2)
        self.assertEqual(self.env.get_action_size(), 4)
    
    def test_state_index_conversion(self):
        """Testa conversão entre estado e índice"""
        # Teste conversão estado -> índice
        index = self.env.get_state_index([1, 2])
        expected_index = 1 * 3 + 2  # row * grid_size + col
        self.assertEqual(index, expected_index)
        
        # Teste conversão índice -> estado
        state = self.env.get_state_from_index(index)
        self.assertEqual(state, [1, 2])
    
    def test_valid_movement(self):
        """Testa movimento válido"""
        # Movimento para direita (ação 3)
        initial_state = self.env.get_state()
        state, reward, done, info = self.env.step(GridWorldAction.RIGHT.value)
        
        # Verificar que se moveu
        self.assertEqual(state, [0, 1])
        self.assertEqual(self.env.current_pos, (0, 1))
        self.assertTrue(info.get('moved', False))
        self.assertFalse(done)
        self.assertEqual(self.env.step_count, 1)
    
    def test_boundary_collision(self):
        """Testa colisão com boundary"""
        # Tentar movimento para cima quando já está no topo
        state, reward, done, info = self.env.step(GridWorldAction.UP.value)
        
        # Não deve ter se movido
        self.assertEqual(state, [0, 0])
        self.assertEqual(self.env.current_pos, (0, 0))
        self.assertEqual(info.get('collision'), 'boundary')
        self.assertEqual(reward, self.env.wall_penalty)
        self.assertFalse(done)
    
    def test_goal_reached(self):
        """Testa chegada ao objetivo"""
        # Mover para próximo do objetivo
        self.env.current_pos = (2, 1)
        
        # Movimento para direita deve alcançar o objetivo
        state, reward, done, info = self.env.step(GridWorldAction.RIGHT.value)
        
        self.assertEqual(state, [2, 2])
        self.assertEqual(reward, self.env.goal_reward)
        self.assertTrue(done)
        self.assertTrue(info.get('goal_reached', False))
    
    def test_wall_collision(self):
        """Testa colisão com parede"""
        # Criar ambiente com parede
        env_with_wall = GridWorld(
            grid_size=3,
            walls=[(1, 1)]
        )
        
        # Mover para posição adjacente à parede
        env_with_wall.current_pos = (1, 0)
        
        # Tentar movimento para direita (para a parede)
        state, reward, done, info = env_with_wall.step(GridWorldAction.RIGHT.value)
        
        # Não deve ter se movido
        self.assertEqual(state, [1, 0])
        self.assertEqual(info.get('collision'), 'wall')
        self.assertEqual(reward, env_with_wall.wall_penalty)
    
    def test_penalty_cell(self):
        """Testa entrada em célula de penalidade"""
        # Criar ambiente com penalidade
        env_with_penalty = GridWorld(
            grid_size=3,
            penalties=[(1, 1)]
        )
        
        # Mover para posição adjacente à penalidade
        env_with_penalty.current_pos = (1, 0)
        
        # Movimento para direita (para a penalidade)
        state, reward, done, info = env_with_penalty.step(GridWorldAction.RIGHT.value)
        
        # Deve ter se movido mas recebido penalidade
        self.assertEqual(state, [1, 1])
        self.assertEqual(reward, env_with_penalty.penalty_reward)
        self.assertTrue(info.get('penalty', False))
    
    def test_max_steps_limit(self):
        """Testa limite máximo de passos"""
        # Criar ambiente com limite baixo
        env = GridWorld(grid_size=3, max_steps=2)
        
        # Fazer dois movimentos
        env.step(GridWorldAction.RIGHT.value)
        state, reward, done, info = env.step(GridWorldAction.DOWN.value)
        
        # Deve ter atingido o limite
        self.assertTrue(done)
        self.assertTrue(info.get('max_steps_reached', False))
        self.assertEqual(env.step_count, 2)
    
    def test_reset(self):
        """Testa reset do ambiente"""
        # Fazer alguns movimentos
        self.env.step(GridWorldAction.RIGHT.value)
        self.env.step(GridWorldAction.DOWN.value)
        
        # Reset
        initial_state = self.env.reset()
        
        # Verificar reset
        self.assertEqual(initial_state, [0, 0])
        self.assertEqual(self.env.current_pos, (0, 0))
        self.assertEqual(self.env.step_count, 0)
        self.assertFalse(self.env.done)
        self.assertEqual(self.env.episode_count, 1)
    
    def test_is_terminal_state(self):
        """Testa detecção de estado terminal"""
        # Estado não terminal
        self.assertFalse(self.env.is_terminal_state([0, 0]))
        self.assertFalse(self.env.is_terminal_state([1, 1]))
        
        # Estado terminal (objetivo)
        self.assertTrue(self.env.is_terminal_state([2, 2]))
    
    def test_get_valid_actions(self):
        """Testa obtenção de ações válidas"""
        # No canto superior esquerdo
        valid_actions = self.env.get_valid_actions([0, 0])
        # Só pode ir para baixo e direita
        self.assertIn(GridWorldAction.DOWN.value, valid_actions)
        self.assertIn(GridWorldAction.RIGHT.value, valid_actions)
        self.assertNotIn(GridWorldAction.UP.value, valid_actions)
        self.assertNotIn(GridWorldAction.LEFT.value, valid_actions)
        
        # No centro
        valid_actions = self.env.get_valid_actions([1, 1])
        # Pode ir para todas as direções
        self.assertEqual(len(valid_actions), 4)
    
    def test_sample_random_action(self):
        """Testa amostragem de ação aleatória"""
        action = self.env.sample_random_action()
        self.assertIn(action, [0, 1, 2, 3])
    
    def test_render_text(self):
        """Testa renderização em texto"""
        rendered = self.env.render('text')
        self.assertIsInstance(rendered, str)
        self.assertIn('Grid World', rendered)
        self.assertIn('A', rendered)  # Agent position
        self.assertIn('G', rendered)  # Goal position
    
    def test_render_symbols(self):
        """Testa renderização com símbolos"""
        rendered = self.env.render('symbols')
        self.assertIsInstance(rendered, str)
        lines = rendered.split('\n')
        self.assertEqual(len(lines), self.env.grid_size)
    
    def test_get_info(self):
        """Testa obtenção de informações do ambiente"""
        info = self.env.get_info()
        
        # Verificar campos obrigatórios
        self.assertIn('grid_size', info)
        self.assertIn('current_pos', info)
        self.assertIn('goal_pos', info)
        self.assertIn('step_count', info)
        self.assertIn('state_size', info)
        self.assertIn('action_size', info)
        
        # Verificar valores
        self.assertEqual(info['grid_size'], 3)
        self.assertEqual(info['state_size'], 2)
        self.assertEqual(info['action_size'], 4)
    
    def test_copy(self):
        """Testa cópia do ambiente"""
        # Configurar ambiente original
        self.env.walls = [(1, 1)]
        self.env.penalties = [(2, 1)]
        
        # Criar cópia
        env_copy = self.env.copy()
        
        # Verificar que a cópia tem as mesmas configurações
        self.assertEqual(env_copy.grid_size, self.env.grid_size)
        self.assertEqual(env_copy.start_pos, self.env.start_pos)
        self.assertEqual(env_copy.goal_pos, self.env.goal_pos)
        self.assertEqual(env_copy.walls, self.env.walls)
        self.assertEqual(env_copy.penalties, self.env.penalties)
        
        # Verificar que são objetos independentes
        self.assertIsNot(env_copy, self.env)
        self.assertIsNot(env_copy.walls, self.env.walls)


class TestGridWorldFactories(unittest.TestCase):
    """Testes para funções de criação de ambientes"""
    
    def test_create_simple_grid(self):
        """Testa criação de grid simples"""
        env = create_simple_grid(4)
        
        self.assertEqual(env.grid_size, 4)
        self.assertEqual(env.start_pos, (0, 0))
        self.assertEqual(env.goal_pos, (3, 3))
        self.assertEqual(len(env.walls), 0)
        self.assertEqual(len(env.penalties), 0)
    
    def test_create_maze_grid(self):
        """Testa criação de grid labirinto"""
        env = create_maze_grid(5)
        
        self.assertEqual(env.grid_size, 5)
        self.assertGreater(len(env.walls), 0)  # Deve ter paredes
        
        # Para grid pequeno não deve ter obstáculos
        small_env = create_maze_grid(3)
        self.assertEqual(len(small_env.walls), 0)
    
    def test_create_random_grid(self):
        """Testa criação de grid aleatório"""
        env = create_random_grid(4, wall_density=0.1, penalty_density=0.1)
        
        self.assertEqual(env.grid_size, 4)
        self.assertEqual(env.start_pos, (0, 0))
        self.assertEqual(env.goal_pos, (3, 3))
        
        # Não deve ter obstáculos nas posições inicial e final
        self.assertNotIn((0, 0), env.walls)
        self.assertNotIn((0, 0), env.penalties)
        self.assertNotIn((3, 3), env.walls)
        self.assertNotIn((3, 3), env.penalties)


if __name__ == '__main__':
    unittest.main()
