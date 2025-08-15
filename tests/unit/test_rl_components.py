"""
Testes para validar Experience Replay Buffer e Políticas de Exploração.
"""

import sys
import os
import tempfile

# Adicionar o diretório src ao path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from reinforcement_learning.experience_replay import (
    ExperienceReplayBuffer, 
    PrioritizedExperienceReplayBuffer,
    Experience,
    create_simple_buffer
)
from reinforcement_learning.policies import (
    EpsilonGreedyPolicy,
    BoltzmannPolicy,
    UCBPolicy,
    GreedyPolicy,
    RandomPolicy,
    create_exploration_policy,
    PolicyComparison
)


def test_experience_replay_buffer():
    """Testa o Experience Replay Buffer"""
    print("=== Testando Experience Replay Buffer ===")
    
    # Criar buffer
    buffer = ExperienceReplayBuffer(max_size=5)
    print(f"Buffer criado: {buffer}")
    
    # Adicionar experiências
    experiences = [
        ([1, 0], 0, 1.0, [0, 1], False),
        ([0, 1], 1, -0.5, [1, 0], False),
        ([1, 0], 0, 0.0, [0, 0], True),
        ([0, 0], 1, 2.0, [1, 1], False),
        ([1, 1], 0, -1.0, [0, 0], True),
    ]
    
    for state, action, reward, next_state, done in experiences:
        buffer.add(state, action, reward, next_state, done)
    
    print(f"Experiências adicionadas. Buffer: {buffer}")
    
    # Testar amostragem
    if buffer.can_sample(3):
        sample = buffer.sample(3)
        print(f"Amostra de 3 experiências:")
        for i, exp in enumerate(sample):
            print(f"  {i+1}: {exp}")
    
    # Testar estatísticas
    stats = buffer.get_statistics()
    print(f"Estatísticas do buffer:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Testar filtros
    positive_rewards = buffer.filter_by_reward(min_reward=0.0)
    print(f"Experiências com recompensa >= 0: {len(positive_rewards)}")
    
    terminal_exps = buffer.filter_terminal_experiences()
    print(f"Experiências terminais: {len(terminal_exps)}")
    
    print("✅ Testes de Experience Replay Buffer concluídos!\n")


def test_prioritized_buffer():
    """Testa o Prioritized Experience Replay Buffer"""
    print("=== Testando Prioritized Buffer ===")
    
    # Criar buffer priorizado
    buffer = PrioritizedExperienceReplayBuffer(max_size=10, alpha=0.6, beta=0.4)
    
    # Adicionar experiências com diferentes prioridades
    high_priority_exp = ([1, 1], 0, 5.0, [0, 0], True)
    low_priority_exp = ([0, 0], 1, 0.1, [1, 0], False)
    
    buffer.add(*high_priority_exp, priority=10.0)
    buffer.add(*low_priority_exp, priority=1.0)
    
    # Adicionar mais experiências
    for i in range(8):
        buffer.add([i % 2, (i+1) % 2], i % 2, float(i), [1, 1], i % 3 == 0)
    
    print(f"Buffer priorizado: {buffer}")
    
    # Testar amostragem priorizada
    if buffer.can_sample(5):
        experiences, indices, weights = buffer.sample(5)
        print("Amostragem priorizada:")
        for i, (exp, idx, weight) in enumerate(zip(experiences, indices, weights)):
            print(f"  {i+1}: índice={idx}, peso={weight:.3f}, reward={exp.reward}")
    
    print("✅ Testes de Prioritized Buffer concluídos!\n")


def test_epsilon_greedy_policy():
    """Testa política Epsilon-Greedy"""
    print("=== Testando Política Epsilon-Greedy ===")
    
    # Criar política
    policy = EpsilonGreedyPolicy(
        epsilon_start=1.0,
        epsilon_end=0.1,
        epsilon_decay=0.9,
        decay_type='exponential'
    )
    
    # Q-values de exemplo
    q_values = [0.1, 0.8, 0.3, 0.6]  # Ação 1 é a melhor
    
    # Testar seleção de ações ao longo do tempo
    action_counts = [0] * len(q_values)
    
    print(f"Política inicial: epsilon = {policy.get_exploration_rate():.3f}")
    
    for step in range(100):
        action = policy.select_action(q_values, step)
        action_counts[action] += 1
        
        if step % 20 == 0:
            policy.update(step)
            print(f"Passo {step}: epsilon = {policy.get_exploration_rate():.3f}")
    
    print(f"Distribuição final de ações: {action_counts}")
    print(f"Melhor ação (1) selecionada: {action_counts[1]} vezes")
    
    print("✅ Testes de Epsilon-Greedy concluídos!\n")


def test_boltzmann_policy():
    """Testa política Boltzmann"""
    print("=== Testando Política Boltzmann ===")
    
    policy = BoltzmannPolicy(
        temperature_start=2.0,
        temperature_end=0.1,
        temperature_decay=0.95
    )
    
    q_values = [1.0, 3.0, 0.5, 2.0]  # Ação 1 é a melhor
    action_counts = [0] * len(q_values)
    
    print(f"Temperatura inicial: {policy.get_exploration_rate():.3f}")
    
    for step in range(100):
        action = policy.select_action(q_values, step)
        action_counts[action] += 1
        
        if step % 25 == 0:
            policy.update(step)
            print(f"Passo {step}: temperatura = {policy.get_exploration_rate():.3f}")
    
    print(f"Distribuição final de ações: {action_counts}")
    
    print("✅ Testes de Boltzmann concluídos!\n")


def test_ucb_policy():
    """Testa política UCB"""
    print("=== Testando Política UCB ===")
    
    policy = UCBPolicy(c=2.0)
    
    q_values = [0.5, 0.7, 0.3, 0.6]
    action_counts = [0] * len(q_values)
    
    print("Simulando UCB...")
    
    for step in range(50):
        action = policy.select_action(q_values, step)
        action_counts[action] += 1
        
        if step % 10 == 0:
            exploration_rate = policy.get_exploration_rate(step)
            print(f"Passo {step}: exploração = {exploration_rate:.3f}")
    
    print(f"Distribuição final de ações: {action_counts}")
    print(f"Contadores internos: {policy.action_counts}")
    
    print("✅ Testes de UCB concluídos!\n")


def test_policy_comparison():
    """Testa comparação entre políticas"""
    print("=== Testando Comparação de Políticas ===")
    
    # Criar diferentes políticas
    comparison = PolicyComparison()
    
    comparison.add_policy("Epsilon-Greedy", EpsilonGreedyPolicy(epsilon_start=0.3, epsilon_decay=0.99))
    comparison.add_policy("Boltzmann", BoltzmannPolicy(temperature_start=1.0, temperature_decay=0.98))
    comparison.add_policy("Greedy", GreedyPolicy())
    comparison.add_policy("Random", RandomPolicy())
    
    # Q-values que mudam ao longo do tempo
    q_values_sequence = [
        [0.1, 0.5, 0.3],  # Ação 1 é melhor
        [0.4, 0.2, 0.6],  # Ação 2 é melhor
        [0.8, 0.1, 0.3],  # Ação 0 é melhor
        [0.2, 0.9, 0.4],  # Ação 1 é melhor
    ]
    
    print("Simulando políticas...")
    
    for step in range(20):
        q_values = q_values_sequence[step % len(q_values_sequence)]
        comparison.simulate_step(q_values, step)
    
    # Analisar resultados
    summary = comparison.get_summary()
    
    print("Resumo da comparação:")
    for policy_name, results in summary.items():
        print(f"\n{policy_name}:")
        for metric, value in results.items():
            if isinstance(value, float):
                print(f"  {metric}: {value:.3f}")
            else:
                print(f"  {metric}: {value}")
    
    print("✅ Testes de Comparação concluídos!\n")


def test_policy_factory():
    """Testa factory de políticas"""
    print("=== Testando Factory de Políticas ===")
    
    # Testar criação de diferentes políticas
    policies = [
        ("epsilon_greedy", {"epsilon_start": 0.8}),
        ("boltzmann", {"temperature_start": 1.5}),
        ("ucb", {"c": 1.5}),
        ("greedy", {}),
        ("random", {}),
    ]
    
    q_values = [0.2, 0.8, 0.4]
    
    for policy_name, kwargs in policies:
        policy = create_exploration_policy(policy_name, **kwargs)
        action = policy.select_action(q_values)
        exploration_rate = policy.get_exploration_rate()
        
        print(f"{policy_name}: ação={action}, exploração={exploration_rate:.3f}")
    
    print("✅ Testes de Factory concluídos!\n")


def test_buffer_persistence():
    """Testa persistência do buffer"""
    print("=== Testando Persistência do Buffer ===")
    
    # Criar e popular buffer
    original_buffer = create_simple_buffer(max_size=5)
    
    for i in range(3):
        original_buffer.add(
            state=[i, i+1],
            action=i % 2,
            reward=float(i),
            next_state=[i+1, i+2],
            done=(i == 2)
        )
    
    print(f"Buffer original: {original_buffer}")
    
    # Salvar em arquivo temporário
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_file = f.name
    
    try:
        original_buffer.save_to_file(temp_file)
        loaded_buffer = ExperienceReplayBuffer.load_from_file(temp_file)
        
        print(f"Buffer carregado: {loaded_buffer}")
        
        # Comparar estatísticas
        orig_stats = original_buffer.get_statistics()
        loaded_stats = loaded_buffer.get_statistics()
        
        print("Comparando estatísticas:")
        for key in orig_stats:
            print(f"  {key}: original={orig_stats[key]}, carregado={loaded_stats[key]}")
        
        # Testar se amostragem funciona
        if loaded_buffer.can_sample(2):
            sample = loaded_buffer.sample(2)
            print(f"Amostra do buffer carregado: {len(sample)} experiências")
        
    finally:
        os.unlink(temp_file)
    
    print("✅ Testes de Persistência concluídos!\n")


if __name__ == "__main__":
    print("🚀 Iniciando testes de Experience Replay e Políticas...\n")
    
    try:
        test_experience_replay_buffer()
        test_prioritized_buffer()
        test_epsilon_greedy_policy()
        test_boltzmann_policy()
        test_ucb_policy()
        test_policy_comparison()
        test_policy_factory()
        test_buffer_persistence()
        
        print("🎉 Todos os testes de RL passaram!")
        print("✅ Experience Replay e Políticas implementadas com sucesso!")
        
    except Exception as e:
        print(f"❌ Erro durante os testes: {e}")
        import traceback
        traceback.print_exc()
