#!/usr/bin/env python3
"""
Testes de aceitação para Agentes de Reinforcement Learning
TDD: Red-Green-Refactor

Este arquivo testa o comportamento end-to-end dos agentes em um ambiente simples.
Simula um episódio completo de treinamento e validação.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from reinforcement_learning.agents import AgentFactory
import random

class SimpleGridWorld:
    """
    Ambiente de teste: Grid World 3x3
    
    Estados: 0-8 (posições no grid 3x3)
    Ações: 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT
    Objetivo: Chegar ao estado 8 (canto inferior direito)
    """
    
    def __init__(self):
        self.size = 3
        self.state = 0  # Posição inicial
        self.goal = 8   # Objetivo
        self.max_steps = 50
        self.step_count = 0
    
    def reset(self):
        """Reseta o ambiente para o estado inicial"""
        self.state = 0
        self.step_count = 0
        return self.state
    
    def step(self, action):
        """
        Executa uma ação no ambiente
        
        Returns:
            next_state, reward, done, info
        """
        row, col = divmod(self.state, self.size)
        
        # Aplicar ação
        if action == 0:  # UP
            row = max(0, row - 1)
        elif action == 1:  # DOWN
            row = min(self.size - 1, row + 1)
        elif action == 2:  # LEFT
            col = max(0, col - 1)
        elif action == 3:  # RIGHT
            col = min(self.size - 1, col + 1)
        
        next_state = row * self.size + col
        self.state = next_state
        self.step_count += 1
        
        # Calcular recompensa
        if next_state == self.goal:
            reward = 10.0  # Recompensa por chegar ao objetivo
            done = True
        elif self.step_count >= self.max_steps:
            reward = -1.0  # Penalidade por timeout
            done = True
        else:
            reward = -0.01  # Pequena penalidade por cada passo
            done = False
        
        return next_state, reward, done, {}
    
    def get_state_vector(self, state):
        """Converte estado para representação vetorial (one-hot)"""
        vector = [0.0] * (self.size * self.size)
        vector[state] = 1.0
        return vector


def test_qlearning_training():
    """Teste de aceitação: Q-Learning em Grid World"""
    print("🎯 Teste de Aceitação: Q-Learning Training")
    
    env = SimpleGridWorld()
    agent = AgentFactory.create_agent(
        agent_type='qlearning',
        state_size=9,
        action_size=4,
        learning_rate=0.1,
        discount_factor=0.95,
        exploration_params={'initial_epsilon': 1.0, 'decay_rate': 0.999, 'min_epsilon': 0.1}
    )
    
    # Estatísticas de treinamento
    episode_rewards = []
    episode_lengths = []
    success_count = 0
    
    # Treinamento
    print("🚀 Iniciando treinamento...")
    for episode in range(100):
        state = env.reset()
        total_reward = 0
        steps = 0
        
        while True:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            
            agent.update(state, action, reward, next_state, done)
            
            total_reward += reward
            steps += 1
            state = next_state
            
            if done:
                if env.state == env.goal:
                    success_count += 1
                break
        
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        
        if episode % 20 == 0:
            avg_reward = sum(episode_rewards[-20:]) / min(20, len(episode_rewards))
            print(f"  Episódio {episode}: reward médio = {avg_reward:.2f}, sucessos = {success_count}")
    
    # Validação
    print("\n🧪 Testando política aprendida...")
    test_successes = 0
    test_episodes = 10
    
    # Definir epsilon para zero (modo greedy) para teste
    agent.policy.current_epsilon = 0.0
    
    for test_ep in range(test_episodes):
        state = env.reset()
        steps = 0
        
        while steps < env.max_steps:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            
            state = next_state
            steps += 1
            
            if done and env.state == env.goal:
                test_successes += 1
                break
            elif done:
                break
    
    success_rate = test_successes / test_episodes
    avg_reward = sum(episode_rewards[-10:]) / 10
    
    print(f"\n📊 Resultados Q-Learning:")
    print(f"  Taxa de sucesso no teste: {success_rate:.1%}")
    print(f"  Recompensa média final: {avg_reward:.2f}")
    print(f"  Episódios de treinamento: {len(episode_rewards)}")
    print(f"  Sucessos no treinamento: {success_count}")
    
    # Mostrar política aprendida
    print(f"\n🎯 Política aprendida (melhores ações por estado):")
    for state in range(9):
        best_action = agent.get_best_action(state)
        q_values = agent.get_q_values(state)
        actions_names = ['UP', 'DOWN', 'LEFT', 'RIGHT']
        print(f"  Estado {state}: {actions_names[best_action]} (Q-values: {[f'{q:.3f}' for q in q_values]})")
    
    assert success_rate >= 0.5, f"Taxa de sucesso muito baixa: {success_rate:.1%}"
    print("✅ Q-Learning passou no teste de aceitação!")


def test_dqn_training():
    """Teste de aceitação: DQN em Grid World"""
    print("\n🎯 Teste de Aceitação: DQN Training")
    
    env = SimpleGridWorld()
    agent = AgentFactory.create_agent(
        agent_type='dqn',
        state_size=9,
        action_size=4,
        hidden_layers=[32, 16],
        learning_rate=0.01,
        discount_factor=0.95,
        exploration_params={'initial_epsilon': 1.0, 'decay_rate': 0.999, 'min_epsilon': 0.1},
        replay_buffer_size=1000,
        batch_size=16,
        target_update_frequency=50
    )
    
    # Estatísticas de treinamento
    episode_rewards = []
    episode_lengths = []
    success_count = 0
    total_losses = []
    
    # Treinamento
    print("🚀 Iniciando treinamento DQN...")
    for episode in range(150):  # DQN precisa de mais episódios
        state_vec = env.get_state_vector(env.reset())
        total_reward = 0
        steps = 0
        episode_loss = 0
        
        while True:
            action = agent.select_action(state_vec)
            next_state, reward, done, _ = env.step(action)
            next_state_vec = env.get_state_vector(next_state)
            
            agent.store_experience(state_vec, action, reward, next_state_vec, done)
            
            # Treinar se houver experiências suficientes
            if agent.can_train():
                loss = agent.train_step()
                episode_loss += loss
                agent.update_target_network()
            
            total_reward += reward
            steps += 1
            state_vec = next_state_vec
            
            if done:
                if env.state == env.goal:
                    success_count += 1
                break
        
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        if episode_loss > 0:
            total_losses.append(episode_loss)
        
        if episode % 30 == 0:
            avg_reward = sum(episode_rewards[-30:]) / min(30, len(episode_rewards))
            avg_loss = sum(total_losses[-10:]) / max(1, len(total_losses[-10:]))
            print(f"  Episódio {episode}: reward médio = {avg_reward:.2f}, loss = {avg_loss:.4f}, sucessos = {success_count}")
    
    # Validação com exploração reduzida mas não zero
    print("\n🧪 Testando política DQN aprendida...")
    test_successes = 0
    test_episodes = 10
    
    # Definir epsilon baixo mas não zero para DQN
    agent.policy.current_epsilon = 0.05  # 5% de exploração ainda
    
    for test_ep in range(test_episodes):
        state_vec = env.get_state_vector(env.reset())
        steps = 0
        
        while steps < env.max_steps:
            action = agent.select_action(state_vec)
            next_state, reward, done, _ = env.step(action)
            
            state_vec = env.get_state_vector(next_state)
            steps += 1
            
            if done and env.state == env.goal:
                test_successes += 1
                break
            elif done:
                break
    
    success_rate = test_successes / test_episodes
    avg_reward = sum(episode_rewards[-10:]) / 10
    avg_loss = sum(total_losses[-10:]) / max(1, len(total_losses[-10:]))
    
    print(f"\n📊 Resultados DQN:")
    print(f"  Taxa de sucesso no teste: {success_rate:.1%}")
    print(f"  Recompensa média final: {avg_reward:.2f}")
    print(f"  Loss média final: {avg_loss:.4f}")
    print(f"  Episódios de treinamento: {len(episode_rewards)}")
    print(f"  Sucessos no treinamento: {success_count}")
    print(f"  Experiências no buffer: {len(agent.replay_buffer)}")
    
    # Mostrar algumas predições
    print(f"\n🎯 Predições DQN para alguns estados:")
    for state in [0, 4, 8]:
        state_vec = env.get_state_vector(state)
        q_values = agent.get_q_values(state_vec)
        actions_names = ['UP', 'DOWN', 'LEFT', 'RIGHT']
        best_action = q_values.index(max(q_values))
        print(f"  Estado {state}: {actions_names[best_action]} (Q-values: {[f'{q:.3f}' for q in q_values]})")
    
    # DQN pode ser mais instável e precisa de um pouco de exploração
    assert success_rate >= 0.2, f"Taxa de sucesso muito baixa para DQN: {success_rate:.1%}"
    print("✅ DQN passou no teste de aceitação!")


def test_agent_comparison():
    """Teste de aceitação: Comparação direta entre agentes"""
    print("\n🎯 Teste de Aceitação: Comparação entre Agentes")
    
    env = SimpleGridWorld()
    
    # Criar agentes
    qlearning_agent = AgentFactory.create_agent(
        'qlearning', 9, 4, learning_rate=0.2,
        exploration_params={'initial_epsilon': 0.8, 'decay_rate': 0.998}
    )
    
    dqn_agent = AgentFactory.create_agent(
        'dqn', 9, 4, hidden_layers=[16],
        exploration_params={'initial_epsilon': 0.8, 'decay_rate': 0.998},
        batch_size=8
    )
    
    agents = {
        'Q-Learning': qlearning_agent,
        'DQN': dqn_agent
    }
    
    results = {}
    
    print("🚀 Treinando agentes em paralelo...")
    
    for name, agent in agents.items():
        print(f"\n  Treinando {name}...")
        successes = 0
        total_reward = 0
        episodes = 80
        
        for episode in range(episodes):
            if name == 'DQN':
                state = env.get_state_vector(env.reset())
            else:
                state = env.reset()
            
            episode_reward = 0
            
            for step in range(50):
                action = agent.select_action(state)
                next_state_raw, reward, done, _ = env.step(action)
                
                if name == 'DQN':
                    next_state = env.get_state_vector(next_state_raw)
                    agent.store_experience(state, action, reward, next_state, done)
                    
                    if agent.can_train():
                        agent.train_step()
                        agent.update_target_network()
                else:
                    next_state = next_state_raw
                    agent.update(state, action, reward, next_state, done)
                
                episode_reward += reward
                state = next_state
                
                if done:
                    if env.state == env.goal:
                        successes += 1
                    break
            
            total_reward += episode_reward
        
        success_rate = successes / episodes
        avg_reward = total_reward / episodes
        
        results[name] = {
            'success_rate': success_rate,
            'avg_reward': avg_reward,
            'episodes': episodes,
            'successes': successes
        }
    
    print(f"\n📊 Comparação Final:")
    for name, result in results.items():
        print(f"  {name}:")
        print(f"    Taxa de sucesso: {result['success_rate']:.1%}")
        print(f"    Recompensa média: {result['avg_reward']:.2f}")
        print(f"    Sucessos: {result['successes']}/{result['episodes']}")
    
    # Pelo menos um agente deve ter desempenho razoável
    best_success_rate = max(r['success_rate'] for r in results.values())
    assert best_success_rate >= 0.3, f"Nenhum agente conseguiu taxa de sucesso >= 30%"
    
    print("✅ Comparação entre agentes passou no teste de aceitação!")


if __name__ == "__main__":
    print("🎯 Executando Testes de Aceitação dos Agentes de RL...\n")
    
    try:
        test_qlearning_training()
        test_dqn_training()
        test_agent_comparison()
        
        print("\n🎉 Todos os testes de aceitação passaram!")
        print("✅ Agentes de RL prontos para produção!")
        
    except Exception as e:
        print(f"\n❌ Teste de aceitação falhou: {e}")
        print("🔧 Agentes precisam de ajustes antes da produção.")
        raise
