#!/usr/bin/env python3
"""
Testes unitários para Agentes de Reinforcement Learning
TDD: Red-Green-Refactor

Este arquivo testa:
- QLearningAgent: Agente básico com Q-table
- DQNAgent: Deep Q-Network com rede neural
- AgentFactory: Factory para criação de agentes
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from reinforcement_learning.agents import QLearningAgent, DQNAgent, AgentFactory
from neural_network.network import NeuralNetwork
import random

def test_qlearning_agent():
    """Testa o agente Q-Learning básico"""
    print("🚀 Iniciando testes do QLearningAgent...")
    
    # Criar agente para ambiente 3x3 grid world
    agent = QLearningAgent(
        state_size=9,  # 3x3 = 9 estados
        action_size=4,  # [up, down, left, right]
        learning_rate=0.1,
        discount_factor=0.95,
        exploration_policy='epsilon_greedy',
        exploration_params={'initial_epsilon': 1.0, 'decay_rate': 0.995, 'min_epsilon': 0.01}
    )
    
    print(f"Agente criado: {agent}")
    print(f"Q-table dimensões: {len(agent.q_table)}x{len(agent.q_table[0])}")
    print(f"Parâmetros: lr={agent.learning_rate}, gamma={agent.discount_factor}")
    
    # Testar seleção de ação
    state = 0  # Estado inicial
    action = agent.select_action(state)
    print(f"Ação selecionada no estado {state}: {action}")
    assert 0 <= action < 4, "Ação deve estar no range válido"
    
    # Testar Q-values
    q_values = agent.get_q_values(state)
    print(f"Q-values para estado {state}: {q_values}")
    assert len(q_values) == 4, "Deve ter Q-value para cada ação"
    
    # Testar update da Q-table
    q_value_before = agent.q_table[state][action]
    agent.update(state, action, reward=1.0, next_state=1, done=False)
    q_value_after = agent.q_table[state][action]
    print(f"Q-value antes: {q_value_before:.4f}, depois: {q_value_after:.4f}")
    
    # Testar episódio completo
    print("\n=== Simulando episódio completo ===")
    episode_steps = []
    state = 0
    for step in range(10):
        action = agent.select_action(state)
        next_state = random.randint(0, 8)
        reward = 1.0 if next_state == 8 else 0.0  # Recompensa no objetivo
        done = (next_state == 8)
        
        agent.update(state, action, reward, next_state, done)
        episode_steps.append((state, action, reward, next_state, done))
        
        if done:
            break
        state = next_state
    
    print(f"Episódio completado em {len(episode_steps)} passos")
    for i, (s, a, r, ns, d) in enumerate(episode_steps):
        print(f"  Step {i}: s={s}, a={a}, r={r}, s'={ns}, done={d}")
    
    # Testar exploração decai
    initial_epsilon = agent.policy.get_exploration_rate()
    for _ in range(100):
        agent.policy.update()
    final_epsilon = agent.policy.get_exploration_rate()
    print(f"Epsilon inicial: {initial_epsilon:.3f}, final: {final_epsilon:.3f}")
    assert final_epsilon < initial_epsilon, "Epsilon deve decair"
    
    print("✅ Testes do QLearningAgent concluídos!")

def test_dqn_agent():
    """Testa o agente DQN (Deep Q-Network)"""
    print("\n🚀 Iniciando testes do DQNAgent...")
    
    # Criar rede neural para DQN
    network = NeuralNetwork(
        layer_sizes=[4, 32, 16, 2],  # Estado 4D, 2 ações
        activations=['relu', 'relu', 'linear'],
        network_name="DQN"
    )
    
    agent = DQNAgent(
        network=network,
        learning_rate=0.001,
        discount_factor=0.99,
        exploration_policy='epsilon_greedy',
        exploration_params={'initial_epsilon': 1.0, 'decay_rate': 0.999, 'min_epsilon': 0.1},
        replay_buffer_size=1000,
        batch_size=32,
        target_update_frequency=100
    )
    
    print(f"Agente DQN criado: {agent}")
    print(f"Rede neural: {network}")
    print(f"Buffer size: {agent.replay_buffer.max_size}")
    
    # Testar predição Q-values
    state = [0.1, -0.5, 1.0, 0.3]
    q_values = agent.get_q_values(state)
    print(f"Q-values para estado {state}: {q_values}")
    assert len(q_values) == 2, "Deve ter Q-value para cada ação"
    
    # Testar seleção de ação
    action = agent.select_action(state)
    print(f"Ação selecionada: {action}")
    assert action in [0, 1], "Ação deve ser válida"
    
    # Adicionar experiências ao buffer
    print("\n=== Adicionando experiências ao buffer ===")
    for i in range(10):
        s = [random.random() for _ in range(4)]
        a = random.choice([0, 1])
        r = random.uniform(-1, 1)
        ns = [random.random() for _ in range(4)]
        done = (i == 9)
        
        agent.store_experience(s, a, r, ns, done)
    
    print(f"Buffer após experiências: {agent.replay_buffer}")
    
    # Testar treinamento
    if agent.can_train():
        loss_before = agent.train_step()
        print(f"Loss do treinamento: {loss_before:.6f}")
    
    # Testar update da target network
    target_updated = agent.update_target_network()
    print(f"Target network atualizada: {target_updated}")
    
    print("✅ Testes do DQNAgent concluídos!")

def test_agent_factory():
    """Testa a factory para criação de agentes"""
    print("\n🚀 Iniciando testes do AgentFactory...")
    
    # Criar agente Q-Learning via factory
    qlearning_agent = AgentFactory.create_agent(
        agent_type='qlearning',
        state_size=6,
        action_size=3,
        learning_rate=0.2,
        discount_factor=0.9
    )
    print(f"Q-Learning Agent: {qlearning_agent}")
    assert qlearning_agent.state_size == 6
    assert qlearning_agent.action_size == 3
    
    # Criar agente DQN via factory
    dqn_agent = AgentFactory.create_agent(
        agent_type='dqn',
        state_size=8,
        action_size=4,
        hidden_layers=[64, 32],
        learning_rate=0.0005
    )
    print(f"DQN Agent: {dqn_agent}")
    assert dqn_agent.network.layer_sizes == [8, 64, 32, 4]
    
    # Testar tipos de agente disponíveis
    available_types = AgentFactory.get_available_types()
    print(f"Tipos disponíveis: {available_types}")
    assert 'qlearning' in available_types
    assert 'dqn' in available_types
    
    print("✅ Testes do AgentFactory concluídos!")

def test_agent_comparison():
    """Testa comparação entre diferentes agentes"""
    print("\n🚀 Iniciando testes de comparação entre agentes...")
    
    # Criar ambiente simples para teste
    def simple_environment(state, action):
        """Ambiente de teste: objetivo é chegar ao estado 5"""
        if action == 0:  # mover para direita
            next_state = min(state + 1, 5)
        elif action == 1:  # mover para esquerda
            next_state = max(state - 1, 0)
        else:  # ficar parado
            next_state = state
        
        reward = 10.0 if next_state == 5 else -0.1
        done = (next_state == 5)
        return next_state, reward, done
    
    # Criar agentes
    qlearning = AgentFactory.create_agent(
        agent_type='qlearning',
        state_size=6,
        action_size=3,
        learning_rate=0.1,
        exploration_params={'initial_epsilon': 0.3}
    )
    
    dqn = AgentFactory.create_agent(
        agent_type='dqn',
        state_size=6,
        action_size=3,
        hidden_layers=[16, 8],
        exploration_params={'initial_epsilon': 0.3}
    )
    
    agents = {'Q-Learning': qlearning, 'DQN': dqn}
    results = {}
    
    # Testar cada agente
    for name, agent in agents.items():
        total_reward = 0
        episodes_solved = 0
        
        for episode in range(10):
            state = 0
            episode_reward = 0
            
            for step in range(20):
                # Converter estado para formato apropriado
                if name == 'DQN':
                    state_vector = [0.0] * 6
                    state_vector[state] = 1.0  # One-hot encoding
                    action = agent.select_action(state_vector)
                    
                    next_state, reward, done = simple_environment(state, action)
                    
                    next_state_vector = [0.0] * 6
                    next_state_vector[next_state] = 1.0
                    
                    agent.store_experience(state_vector, action, reward, next_state_vector, done)
                    if agent.can_train():
                        agent.train_step()
                else:
                    action = agent.select_action(state)
                    next_state, reward, done = simple_environment(state, action)
                    agent.update(state, action, reward, next_state, done)
                
                episode_reward += reward
                state = next_state
                
                if done:
                    episodes_solved += 1
                    break
            
            total_reward += episode_reward
        
        results[name] = {
            'total_reward': total_reward,
            'episodes_solved': episodes_solved,
            'avg_reward': total_reward / 10
        }
    
    # Mostrar resultados
    print("Resultados da comparação:")
    for name, result in results.items():
        print(f"  {name}:")
        print(f"    Recompensa total: {result['total_reward']:.2f}")
        print(f"    Episódios resolvidos: {result['episodes_solved']}/10")
        print(f"    Recompensa média: {result['avg_reward']:.2f}")
    
    print("✅ Testes de comparação concluídos!")

if __name__ == "__main__":
    print("🎯 Executando testes dos Agentes de RL com TDD...\n")
    
    test_qlearning_agent()
    test_dqn_agent()
    test_agent_factory()
    test_agent_comparison()
    
    print("\n🎉 Todos os testes dos agentes passaram!")
    print("✅ Agentes de RL implementados com sucesso!")
