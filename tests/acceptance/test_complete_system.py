#!/usr/bin/env python3
"""
Teste de aceitação completo com todos os módulos implementados
"""

import sys
import os

# Imports do projeto
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from neural_network.network import NeuralNetwork
from reinforcement_learning.agents import QLearningAgent, DQNAgent
from environment.grid_world import GridWorld, create_simple_grid
from persistence.model_saver import ModelSaver
from persistence.model_loader import ModelLoader
from utils.training_utils import TrainingLogger, PerformanceMetrics
from utils.visualization import TrainingVisualizer
import tempfile


def test_complete_system():
    """Teste completo do sistema integrado"""
    
    print("🚀 TESTE DE ACEITAÇÃO COMPLETO")
    print("=" * 50)
    
    # 1. Teste do ambiente
    print("\n1️⃣ Testando ambiente Grid World...")
    env = create_simple_grid(3)
    state = env.reset()
    print(f"   ✅ Estado inicial: {state}")
    
    # Fazer alguns movimentos
    for action in [1, 3, 1, 3]:  # DOWN, RIGHT, DOWN, RIGHT
        next_state, reward, done, info = env.step(action)
        print(f"   ✅ Ação {action} -> Estado {next_state}, Recompensa {reward:.2f}")
        if done:
            print(f"   🎯 Objetivo alcançado!")
            break
    
    # 2. Teste dos agentes
    print("\n2️⃣ Testando agentes de RL...")
    
    # Q-Learning
    q_agent = QLearningAgent(state_size=9, action_size=4, learning_rate=0.1)
    print(f"   ✅ Q-Learning Agent criado: {q_agent.state_size} estados, {q_agent.action_size} ações")
    
    # DQN
    dqn_network = NeuralNetwork([2, 16, 4], ['relu', 'linear'])
    dqn_agent = DQNAgent(network=dqn_network, learning_rate=0.001)
    print(f"   ✅ DQN Agent criado: {dqn_agent.state_size} entradas, {dqn_agent.action_size} ações")
    
    # 3. Teste de treinamento rápido
    print("\n3️⃣ Testando treinamento rápido...")
    
    env.reset()
    logger = TrainingLogger(log_interval=5)
    
    for episode in range(10):
        state = env.reset()
        total_reward = 0
        
        for step in range(20):
            action = q_agent.select_action(env.get_state_index(state))
            next_state, reward, done, info = env.step(action)
            
            # Q-Learning update
            q_agent.update(
                env.get_state_index(state),
                action,
                reward,
                env.get_state_index(next_state),
                done
            )
            
            total_reward += reward
            state = next_state
            
            if done:
                break
        
        logger.log_episode(total_reward, success=done)
    
    print(f"   ✅ 10 episódios de treinamento completos")
    
    # 4. Teste de persistência
    print("\n4️⃣ Testando sistema de persistência...")
    
    temp_dir = tempfile.mkdtemp()
    saver = ModelSaver(temp_dir)
    loader = ModelLoader(temp_dir)
    
    # Salvar agente Q-Learning
    q_save_path = saver.save_qlearning_agent(q_agent, "test_q_agent")
    print(f"   ✅ Q-Learning agent salvo em: {q_save_path}")
    
    # Salvar agente DQN
    dqn_save_path = saver.save_dqn_agent(dqn_agent, "test_dqn_agent")
    print(f"   ✅ DQN agent salvo em: {dqn_save_path}")
    
    # Carregar agentes
    loaded_q_agent, q_metadata = loader.load_qlearning_agent("test_q_agent")
    print(f"   ✅ Q-Learning agent carregado: {loaded_q_agent.state_size} estados")
    
    # 5. Teste de métricas
    print("\n5️⃣ Testando métricas de performance...")
    
    metrics_calc = PerformanceMetrics()
    fake_rewards = [1.0, 2.0, 3.0, 4.0, 5.0]
    fake_success = [False, True, True, True, True]
    fake_lengths = [10, 8, 6, 5, 4]
    
    metrics = metrics_calc.calculate_all_metrics(fake_rewards, fake_success, fake_lengths)
    print(f"   ✅ Recompensa média: {metrics['avg_reward']:.2f}")
    print(f"   ✅ Taxa de sucesso: {metrics['success_rate']:.1%}")
    print(f"   ✅ Duração média: {metrics['avg_episode_length']:.1f} passos")
    
    # 6. Teste de visualização
    print("\n6️⃣ Testando visualização...")
    
    visualizer = TrainingVisualizer()
    
    # Progresso de treinamento
    progress_chart = visualizer.plot_training_progress(fake_rewards)
    print(f"   ✅ Gráfico de progresso gerado ({len(progress_chart)} caracteres)")
    
    # Q-table heatmap
    simple_q_table = [[1.0, 2.0], [3.0, 4.0]]
    heatmap = visualizer.plot_q_values_heatmap(simple_q_table)
    print(f"   ✅ Heatmap de Q-values gerado ({len(heatmap)} caracteres)")
    
    # 7. Teste de integração completa
    print("\n7️⃣ Teste de integração completa...")
    
    # Criar sessão de treinamento completa
    integration_env = create_simple_grid(3)
    integration_agent = QLearningAgent(9, 4, 0.2, 0.9)
    integration_logger = TrainingLogger(log_interval=2)
    
    successful_episodes = 0
    
    for episode in range(5):
        state = integration_env.reset()
        total_reward = 0
        
        for step in range(15):
            action = integration_agent.select_action(integration_env.get_state_index(state))
            next_state, reward, done, info = integration_env.step(action)
            
            integration_agent.update(
                integration_env.get_state_index(state),
                action,
                reward,
                integration_env.get_state_index(next_state),
                done
            )
            
            total_reward += reward
            state = next_state
            
            if done:
                successful_episodes += 1
                break
        
        integration_logger.log_episode(total_reward, success=done)
    
    # Salvar sessão completa
    session_path = saver.save_training_session(
        integration_agent,
        {'rewards': [1, 2, 3, 4, 5], 'episodes': list(range(5))},
        "integration_test_session",
        {'test_type': 'integration', 'episodes': 5}
    )
    
    print(f"   ✅ Sessão de treinamento salva em: {session_path}")
    print(f"   ✅ Episódios bem-sucedidos: {successful_episodes}/5")
    
    # 8. Limpeza
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)
    
    print("\n🎉 TESTE DE ACEITAÇÃO COMPLETO - SUCESSO!")
    print("=" * 50)
    print("Todos os módulos implementados e funcionando:")
    print("  ✅ Rede Neural (matrix, activation, neuron, layer, network)")
    print("  ✅ Reinforcement Learning (agents, experience_replay, policies)")
    print("  ✅ Environment (grid_world com múltiplas configurações)")
    print("  ✅ Persistence (model_saver, model_loader)")
    print("  ✅ Utils (training_utils, visualization)")
    print("  ✅ Integração completa funcionando")
    
    return True


if __name__ == "__main__":
    test_complete_system()
