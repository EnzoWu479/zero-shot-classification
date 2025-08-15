# Exemplo de Projeto Usando deeprl-neural

Este diretório contém exemplos de como usar a biblioteca `deeprl-neural` em projetos externos.

## Estrutura de Projeto Exemplo

```
meu_projeto_rl/
├── requirements.txt
├── main.py
├── config.py
└── experiments/
    ├── q_learning_experiment.py
    ├── dqn_experiment.py
    └── neural_network_experiment.py
```

## 1. requirements.txt

```txt
# Se instalada localmente
-e /caminho/para/deeprl-neural

# Ou se disponível via PyPI (futuro)
# deeprl-neural>=1.0.0

# Dependências opcionais para visualização
matplotlib>=3.5.0
numpy>=1.20.0
```

## 2. config.py

```python
"""
Configurações do projeto
"""

# Configurações de treinamento
TRAINING_CONFIG = {
    'episodes': 1000,
    'max_steps': 100,
    'learning_rate': 0.1,
    'discount_factor': 0.95,
    'epsilon_start': 1.0,
    'epsilon_end': 0.01,
    'epsilon_decay': 0.995
}

# Configurações do ambiente
ENV_CONFIG = {
    'grid_size': 5,
    'start_pos': (0, 0),
    'goal_pos': (4, 4),
    'walls': [(1, 1), (1, 2), (2, 1), (3, 3)],
    'penalties': [(2, 2), (3, 1)]
}

# Configurações da rede neural
NETWORK_CONFIG = {
    'hidden_layers': [64, 32],
    'activation': 'relu',
    'output_activation': 'linear'
}
```

## 3. main.py

```python
"""
Projeto principal usando deeprl-neural
"""

from deeprl_neural import (
    GridWorld, QLearningAgent, DQNAgent, NeuralNetwork,
    TrainingLogger, TrainingVisualizer
)
from config import TRAINING_CONFIG, ENV_CONFIG, NETWORK_CONFIG

def setup_environment():
    """Configura o ambiente de treino"""
    return GridWorld(
        grid_size=ENV_CONFIG['grid_size'],
        start_pos=ENV_CONFIG['start_pos'],
        goal_pos=ENV_CONFIG['goal_pos'],
        walls=ENV_CONFIG['walls'],
        penalties=ENV_CONFIG['penalties']
    )

def train_qlearning_agent(env):
    """Treina um agente Q-Learning"""
    print("🎯 Treinando agente Q-Learning...")
    
    agent = QLearningAgent(
        state_size=env.grid_size * env.grid_size,
        action_size=env.get_action_size(),
        learning_rate=TRAINING_CONFIG['learning_rate'],
        discount_factor=TRAINING_CONFIG['discount_factor']
    )
    
    logger = TrainingLogger()
    
    for episode in range(TRAINING_CONFIG['episodes']):
        env.reset()
        state_index = env.get_state_index()
        total_reward = 0
        steps = 0
        
        while steps < TRAINING_CONFIG['max_steps']:
            action = agent.select_action(state_index)
            next_state, reward, done, info = env.step(action)
            next_state_index = env.get_state_index(next_state)
            
            agent.update(state_index, action, reward, next_state_index, done)
            
            state_index = next_state_index
            total_reward += reward
            steps += 1
            
            if done:
                break
        
        logger.log_episode(episode, total_reward, steps, done and info.get('success', False))
        
        # Log de progresso
        if (episode + 1) % 100 == 0:
            perf = logger.get_recent_performance(100)
            print(f"Episódio {episode + 1}: Taxa de sucesso = {perf['success_rate']:.1%}")
    
    return agent, logger

def train_dqn_agent(env):
    """Treina um agente DQN"""
    print("🧠 Treinando agente DQN...")
    
    # Criar rede neural
    input_size = env.get_state_size()
    output_size = env.get_action_size()
    hidden_layers = NETWORK_CONFIG['hidden_layers']
    
    layers = [input_size] + hidden_layers + [output_size]
    network = NeuralNetwork(layers)
    
    agent = DQNAgent(
        network=network,
        learning_rate=TRAINING_CONFIG['learning_rate']
    )
    
    logger = TrainingLogger()
    
    for episode in range(TRAINING_CONFIG['episodes']):
        state = env.reset()
        total_reward = 0
        steps = 0
        
        while steps < TRAINING_CONFIG['max_steps']:
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            
            # Armazenar experiência
            agent.store_experience(state, action, reward, next_state, done)
            
            # Treinar se houver experiências suficientes
            if agent.can_train():
                agent.update(state, action, reward, next_state, done)
            
            state = next_state
            total_reward += reward
            steps += 1
            
            if done:
                break
        
        logger.log_episode(episode, total_reward, steps, done and info.get('success', False))
        
        # Log de progresso
        if (episode + 1) % 100 == 0:
            perf = logger.get_recent_performance(100)
            print(f"Episódio {episode + 1}: Taxa de sucesso = {perf['success_rate']:.1%}")
    
    return agent, logger

def compare_agents():
    """Compara o desempenho dos agentes"""
    print("\n📊 Comparando Agentes...")
    print("=" * 50)
    
    env = setup_environment()
    
    # Treinar Q-Learning
    q_agent, q_logger = train_qlearning_agent(env)
    q_performance = q_logger.get_recent_performance(100)
    
    # Resetar ambiente para DQN
    env = setup_environment()
    
    # Treinar DQN
    dqn_agent, dqn_logger = train_dqn_agent(env)
    dqn_performance = dqn_logger.get_recent_performance(100)
    
    # Mostrar resultados
    print("\n🏆 Resultados Finais:")
    print(f"Q-Learning: {q_performance['success_rate']:.1%} de sucesso, "
          f"recompensa média: {q_performance['avg_reward']:.2f}")
    print(f"DQN:        {dqn_performance['success_rate']:.1%} de sucesso, "
          f"recompensa média: {dqn_performance['avg_reward']:.2f}")

def test_neural_network():
    """Testa funcionalidades da rede neural"""
    print("\n🧪 Testando Rede Neural...")
    
    from deeprl_neural import Matrix
    
    # Criar rede simples
    nn = NeuralNetwork([3, 5, 2])
    
    # Dados de teste
    inputs = Matrix([[1.0, 0.5, -0.2]])
    
    # Forward pass
    output = nn.forward(inputs)
    print(f"Input: {inputs.data[0]}")
    print(f"Output: {[round(x, 4) for x in output.data[0]]}")
    
    # Teste de treinamento (exemplo básico)
    target = Matrix([[1.0, 0.0]])
    nn.backward(output, target)
    
    print("✅ Rede neural funcionando corretamente!")

def main():
    """Função principal do projeto"""
    print("🚀 Projeto de Reinforcement Learning")
    print("🔧 Usando biblioteca deeprl-neural")
    print("=" * 50)
    
    try:
        # Verificar instalação
        import deeprl_neural
        print(f"✅ deeprl-neural v{deeprl_neural.__version__} carregado")
        
        # Executar experimentos
        test_neural_network()
        compare_agents()
        
        print("\n🎉 Projeto executado com sucesso!")
        
    except ImportError as e:
        print(f"❌ Erro de importação: {e}")
        print("Certifique-se de que deeprl-neural está instalado:")
        print("pip install -e /caminho/para/deeprl-neural")

if __name__ == "__main__":
    main()
```

## 4. experiments/q_learning_experiment.py

```python
"""
Experimento específico com Q-Learning
"""

from deeprl_neural import GridWorld, QLearningAgent, TrainingLogger
import time

def hyperparameter_tuning():
    """Teste de diferentes hiperparâmetros"""
    print("🔬 Experimento: Ajuste de Hiperparâmetros Q-Learning")
    
    env = GridWorld(grid_size=4)
    learning_rates = [0.01, 0.1, 0.3, 0.5]
    results = {}
    
    for lr in learning_rates:
        print(f"\n🧪 Testando learning_rate = {lr}")
        
        agent = QLearningAgent(
            state_size=16,
            action_size=4,
            learning_rate=lr,
            discount_factor=0.9
        )
        
        logger = TrainingLogger()
        start_time = time.time()
        
        # Treinar por 500 episódios
        for episode in range(500):
            env.reset()
            state = env.get_state_index()
            total_reward = 0
            
            for step in range(50):
                action = agent.select_action(state)
                next_state, reward, done, info = env.step(action)
                next_state_idx = env.get_state_index(next_state)
                
                agent.update(state, action, reward, next_state_idx, done)
                
                state = next_state_idx
                total_reward += reward
                
                if done:
                    break
            
            logger.log_episode(episode, total_reward, step + 1, done)
        
        # Avaliar performance
        performance = logger.get_recent_performance(100)
        training_time = time.time() - start_time
        
        results[lr] = {
            'success_rate': performance['success_rate'],
            'avg_reward': performance['avg_reward'],
            'training_time': training_time
        }
        
        print(f"  Sucesso: {performance['success_rate']:.1%}")
        print(f"  Recompensa: {performance['avg_reward']:.2f}")
        print(f"  Tempo: {training_time:.1f}s")
    
    # Mostrar melhor resultado
    best_lr = max(results.keys(), key=lambda x: results[x]['success_rate'])
    print(f"\n🏆 Melhor learning_rate: {best_lr}")
    print(f"   Sucesso: {results[best_lr]['success_rate']:.1%}")

if __name__ == "__main__":
    hyperparameter_tuning()
```

## Como Usar

1. **Instalar a biblioteca** (ver INSTALLATION_GUIDE.md)

2. **Criar projeto:**
```bash
mkdir meu_projeto_rl
cd meu_projeto_rl
```

3. **Copiar os arquivos exemplo acima**

4. **Executar:**
```bash
python main.py
```

## Comandos CLI Disponíveis

Após instalar a biblioteca, você também pode usar:

```bash
# Ver demonstração
deeprl-demo

# Treinar modelos
deeprl-train --agent qlearning --episodes 1000

# Executar testes
deeprl-test --component all
```
