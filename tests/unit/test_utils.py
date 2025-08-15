#!/usr/bin/env python3
"""
Testes para o módulo de utils
"""

import unittest
import sys
import os
import tempfile

# Imports do projeto
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from utils.training_utils import (
    TrainingLogger, EarlyStopping, LearningRateScheduler, 
    RewardShaper, ExperienceBuffer, PerformanceMetrics,
    calculate_moving_average, normalize_rewards, calculate_success_rate,
    epsilon_schedule, create_training_curriculum
)
from utils.visualization import (
    SimpleChart, TrainingVisualizer, print_progress_bar,
    format_training_stats, create_comparison_table,
    calculate_trend, detect_convergence
)


class TestTrainingLogger(unittest.TestCase):
    """Testes para TrainingLogger"""
    
    def setUp(self):
        """Setup para cada teste"""
        self.logger = TrainingLogger(log_interval=5)
    
    def test_init(self):
        """Testa inicialização do logger"""
        self.assertEqual(self.logger.log_interval, 5)
        self.assertEqual(self.logger.current_episode, 0)
        self.assertIn('reward', self.logger.history)
        self.assertIn('loss', self.logger.history)
    
    def test_log_episode(self):
        """Testa logging de episódio"""
        # Log alguns episódios (não devem ser salvos até interval)
        for i in range(4):
            self.logger.log_episode(
                reward=10.0 + i,
                loss=1.0 - i * 0.1,
                epsilon=0.9 - i * 0.1,
                steps=20 + i,
                success=i % 2 == 0
            )
        
        # Ainda não deve ter salvado nada
        self.assertEqual(len(self.logger.history['reward']), 0)
        
        # 5º episódio deve ser salvo
        self.logger.log_episode(15.0, 0.5, 0.4, 25, True)
        
        # Agora deve ter 1 entrada
        self.assertEqual(len(self.logger.history['reward']), 1)
        self.assertEqual(self.logger.history['reward'][0], 15.0)
        self.assertEqual(self.logger.current_episode, 5)
    
    def test_get_recent_performance(self):
        """Testa obtenção de performance recente"""
        # Simular vários episódios
        for i in range(20):
            self.logger.log_episode(
                reward=10.0 + i,
                success=(i > 10)  # Melhoria ao longo do tempo
            )
        
        stats = self.logger.get_recent_performance(window=10)
        
        self.assertIn('avg_reward', stats)
        self.assertIn('success_rate', stats)
        self.assertIn('max_reward', stats)
        self.assertIn('min_reward', stats)
        
        # Verificar que success_rate melhorou
        self.assertGreater(stats['success_rate'], 0.0)


class TestEarlyStopping(unittest.TestCase):
    """Testes para EarlyStopping"""
    
    def test_early_stopping_max_mode(self):
        """Testa early stopping em modo maximização"""
        early_stop = EarlyStopping(patience=3, mode='max')
        
        # Primeiro score define baseline
        self.assertFalse(early_stop.check(1.0))
        
        # Score melhor - reset counter
        self.assertFalse(early_stop.check(2.0))
        
        # Scores piores - incrementa counter
        self.assertFalse(early_stop.check(1.5))  # counter = 1
        self.assertFalse(early_stop.check(1.0))  # counter = 2
        self.assertFalse(early_stop.check(0.5))  # counter = 3
        self.assertTrue(early_stop.check(0.0))   # counter >= patience, deve parar
    
    def test_early_stopping_min_mode(self):
        """Testa early stopping em modo minimização"""
        early_stop = EarlyStopping(patience=2, mode='min')
        
        # Primeiro score define baseline
        self.assertFalse(early_stop.check(3.0))
        
        # Score melhor - reset counter
        self.assertFalse(early_stop.check(2.0))
        
        # Scores piores - incrementa counter
        self.assertFalse(early_stop.check(2.5))  # counter = 1
        self.assertFalse(early_stop.check(3.0))  # counter = 2
        self.assertTrue(early_stop.check(3.5))   # counter >= patience, deve parar


class TestLearningRateScheduler(unittest.TestCase):
    """Testes para LearningRateScheduler"""
    
    def test_constant_schedule(self):
        """Testa agenda constante"""
        scheduler = LearningRateScheduler(0.01, 'constant')
        
        # Deve sempre retornar o mesmo valor
        self.assertEqual(scheduler.get_lr(), 0.01)
        self.assertEqual(scheduler.get_lr(), 0.01)
    
    def test_linear_schedule(self):
        """Testa agenda linear"""
        scheduler = LearningRateScheduler(0.01, 'linear')
        
        # Testar decaimento linear
        lr_start = scheduler.get_lr(episode=0, total_episodes=100)
        lr_mid = scheduler.get_lr(episode=50, total_episodes=100)
        lr_end = scheduler.get_lr(episode=100, total_episodes=100)
        
        self.assertEqual(lr_start, 0.01)
        self.assertLess(lr_mid, lr_start)
        self.assertLess(lr_end, lr_mid)
    
    def test_exponential_schedule(self):
        """Testa agenda exponencial"""
        scheduler = LearningRateScheduler(0.01, 'exponential')
        
        lr1 = scheduler.get_lr()
        lr2 = scheduler.get_lr()
        
        # Deve decair exponencialmente
        self.assertLess(lr2, lr1)


class TestRewardShaper(unittest.TestCase):
    """Testes para RewardShaper"""
    
    def test_no_shaping(self):
        """Testa ausência de shaping"""
        shaper = RewardShaper('none')
        
        original_reward = 5.0
        shaped_reward = shaper.shape_reward([0, 0], [1, 1], original_reward, [2, 2])
        
        self.assertEqual(shaped_reward, original_reward)
    
    def test_distance_shaping(self):
        """Testa shaping baseado em distância"""
        shaper = RewardShaper('distance')
        
        # Movimento que aproxima do objetivo deve dar recompensa extra
        shaped_reward = shaper.shape_reward([0, 0], [1, 1], 0.0, [2, 2])
        self.assertGreater(shaped_reward, 0.0)
        
        # Movimento que afasta do objetivo deve dar penalidade
        shaped_reward = shaper.shape_reward([1, 1], [0, 0], 0.0, [2, 2])
        self.assertLess(shaped_reward, 0.0)


class TestExperienceBuffer(unittest.TestCase):
    """Testes para ExperienceBuffer"""
    
    def test_add_and_sample(self):
        """Testa adição e amostragem"""
        buffer = ExperienceBuffer(capacity=10)
        
        # Adicionar experiências
        for i in range(5):
            experience = {
                'state': [i, i],
                'action': i % 4,
                'reward': float(i),
                'next_state': [i+1, i+1],
                'done': i == 4
            }
            buffer.add(experience)
        
        self.assertEqual(buffer.size(), 5)
        
        # Amostrar
        sample = buffer.sample(3)
        self.assertEqual(len(sample), 3)
        
        # Amostra maior que buffer
        sample = buffer.sample(10)
        self.assertEqual(len(sample), 5)
    
    def test_capacity_limit(self):
        """Testa limite de capacidade"""
        buffer = ExperienceBuffer(capacity=3)
        
        # Adicionar mais que a capacidade
        for i in range(5):
            buffer.add({'value': i})
        
        # Deve ter apenas 3 elementos (os mais recentes)
        self.assertEqual(buffer.size(), 3)
        
        recent = buffer.get_recent(3)
        values = [exp['value'] for exp in recent]
        self.assertEqual(values, [2, 3, 4])


class TestUtilityFunctions(unittest.TestCase):
    """Testes para funções utilitárias"""
    
    def test_calculate_moving_average(self):
        """Testa cálculo de média móvel"""
        data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        moving_avg = calculate_moving_average(data, window=3)
        
        # Primeiros valores devem ser iguais aos originais
        self.assertEqual(moving_avg[0], 1)
        self.assertEqual(moving_avg[1], 2)
        
        # Terceiro valor deve ser média dos primeiros 3
        self.assertEqual(moving_avg[2], (1 + 2 + 3) / 3)
    
    def test_normalize_rewards(self):
        """Testa normalização de recompensas"""
        rewards = [1, 2, 3, 4, 5]
        normalized = normalize_rewards(rewards)
        
        # Média deve ser aproximadamente 0
        mean_normalized = sum(normalized) / len(normalized)
        self.assertAlmostEqual(mean_normalized, 0.0, places=10)
    
    def test_calculate_success_rate(self):
        """Testa cálculo de taxa de sucesso"""
        results = [True, True, False, True, False]
        success_rate = calculate_success_rate(results)
        
        self.assertEqual(success_rate, 0.6)  # 3/5
    
    def test_epsilon_schedule(self):
        """Testa agenda de epsilon"""
        # Linear schedule
        eps_start = epsilon_schedule(0, 100, 1.0, 0.1, 'linear')
        eps_mid = epsilon_schedule(50, 100, 1.0, 0.1, 'linear')
        eps_end = epsilon_schedule(100, 100, 1.0, 0.1, 'linear')
        
        self.assertEqual(eps_start, 1.0)
        self.assertEqual(eps_end, 0.1)
        self.assertGreater(eps_mid, eps_end)
        self.assertLess(eps_mid, eps_start)


class TestVisualization(unittest.TestCase):
    """Testes para módulo de visualização"""
    
    def test_simple_chart_line(self):
        """Testa gráfico de linha simples"""
        chart = SimpleChart(width=20, height=10)
        data = [1, 2, 3, 2, 1]
        
        plot = chart.plot_line(data, "Test Chart")
        
        self.assertIsInstance(plot, str)
        self.assertIn("Test Chart", plot)
        self.assertIn("*", plot)  # Deve ter pontos plotados
    
    def test_simple_chart_histogram(self):
        """Testa histograma simples"""
        chart = SimpleChart(width=20, height=10)
        data = [1, 1, 2, 2, 2, 3, 3, 4, 5]
        
        histogram = chart.plot_histogram(data, bins=5)
        
        self.assertIsInstance(histogram, str)
        self.assertIn("Histogram", histogram)
    
    def test_training_visualizer(self):
        """Testa visualizador de treinamento"""
        visualizer = TrainingVisualizer()
        
        rewards = [1, 2, 3, 4, 5]
        losses = [5, 4, 3, 2, 1]
        
        progress_plot = visualizer.plot_training_progress(rewards, losses)
        
        self.assertIsInstance(progress_plot, str)
        self.assertIn("Training Rewards", progress_plot)
        self.assertIn("Training Loss", progress_plot)
    
    def test_q_values_heatmap(self):
        """Testa mapa de calor de Q-values"""
        visualizer = TrainingVisualizer()
        
        q_table = [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0]
        ]
        
        heatmap = visualizer.plot_q_values_heatmap(
            q_table,
            state_labels=['S1', 'S2', 'S3'],
            action_labels=['A1', 'A2', 'A3']
        )
        
        self.assertIsInstance(heatmap, str)
        self.assertIn("Q-Values Heatmap", heatmap)
        self.assertIn("S1", heatmap)
        self.assertIn("A1", heatmap)
    
    def test_progress_bar(self):
        """Testa barra de progresso"""
        bar = print_progress_bar(50, 100, width=20, prefix="Progress", suffix="Complete")
        
        self.assertIsInstance(bar, str)
        self.assertIn("Progress", bar)
        self.assertIn("50.0%", bar)
        self.assertIn("█", bar)
        self.assertIn("░", bar)
    
    def test_format_training_stats(self):
        """Testa formatação de estatísticas"""
        stats = {
            'avg_reward': 15.5,
            'success_rate': 0.85,
            'total_episodes': 100
        }
        
        formatted = format_training_stats(stats)
        
        self.assertIsInstance(formatted, str)
        self.assertIn("15.5000", formatted)
        self.assertIn("85.0%", formatted)
        self.assertIn("100", formatted)
    
    def test_comparison_table(self):
        """Testa tabela comparativa"""
        agents_data = [
            {'name': 'Agent1', 'avg_reward': 10.5, 'success_rate': 0.8},
            {'name': 'Agent2', 'avg_reward': 12.3, 'success_rate': 0.9}
        ]
        
        table = create_comparison_table(agents_data)
        
        self.assertIsInstance(table, str)
        self.assertIn("Agent1", table)
        self.assertIn("Agent2", table)
        self.assertIn("80.0%", table)
    
    def test_trend_calculation(self):
        """Testa cálculo de tendência"""
        # Dados crescentes (mais de 20 valores para garantir window suficiente)
        increasing_data = list(range(1, 21))  # [1, 2, 3, ..., 20]
        trend = calculate_trend(increasing_data)
        self.assertIn("Improving", trend)
        
        # Dados decrescentes
        decreasing_data = list(range(20, 0, -1))  # [20, 19, 18, ..., 1]
        trend = calculate_trend(decreasing_data)
        self.assertIn("Declining", trend)
    
    def test_convergence_detection(self):
        """Testa detecção de convergência"""
        # Dados que convergem
        converged_data = [1.0] * 100
        self.assertTrue(detect_convergence(converged_data, threshold=0.01, window=50))
        
        # Dados que não convergem
        varying_data = [i % 10 for i in range(100)]
        self.assertFalse(detect_convergence(varying_data, threshold=0.01, window=50))


class TestPerformanceMetrics(unittest.TestCase):
    """Testes para PerformanceMetrics"""
    
    def test_calculate_all_metrics(self):
        """Testa cálculo de todas as métricas"""
        metrics_calc = PerformanceMetrics()
        
        rewards = [1, 2, 3, 4, 5]
        success_flags = [True, True, False, True, False]
        episode_lengths = [10, 15, 20, 12, 18]
        
        metrics = metrics_calc.calculate_all_metrics(rewards, success_flags, episode_lengths)
        
        # Verificar métricas básicas
        self.assertIn('avg_reward', metrics)
        self.assertIn('max_reward', metrics)
        self.assertIn('min_reward', metrics)
        self.assertIn('success_rate', metrics)
        self.assertIn('avg_episode_length', metrics)
        
        # Verificar valores
        self.assertEqual(metrics['avg_reward'], 3.0)
        self.assertEqual(metrics['max_reward'], 5)
        self.assertEqual(metrics['min_reward'], 1)
        self.assertEqual(metrics['success_rate'], 0.6)
        self.assertEqual(metrics['avg_episode_length'], 15.0)


if __name__ == '__main__':
    unittest.main()
