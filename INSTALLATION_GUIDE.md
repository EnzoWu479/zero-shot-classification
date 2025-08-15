# 📦 Guia de Instalação - deeprl-neural

Este guia mostra como instalar e usar a biblioteca `deeprl-neural` em outros projetos.

## 🚀 Métodos de Instalação

### Método 1: Instalação Local (Desenvolvimento)

Se você tem acesso ao código fonte da biblioteca:

```bash
# 1. Clone ou copie o diretório da biblioteca
# 2. Navegue até o diretório do projeto onde quer usar a biblioteca
cd meu-projeto

# 3. Instale a biblioteca em modo de desenvolvimento
pip install -e /caminho/para/deeprl-neural
# ou
pip install -e c:\projetos\testes\zero-shot-classification
```

### Método 2: Instalação via Wheel (Recomendado)

```bash
# 1. No diretório da biblioteca, gere o wheel
cd c:\projetos\testes\zero-shot-classification
python -m build

# 2. No seu projeto, instale o wheel gerado
pip install /caminho/para/deeprl-neural/dist/deeprl_neural-1.0.0-py3-none-any.whl
```

### Método 3: Instalação Direta do Código

```bash
# Se você tem o código fonte
pip install /caminho/para/deeprl-neural
```

### Método 4: Para Projetos com UV

```bash
# No seu projeto que usa UV
cd meu-projeto

# Adicionar como dependência local
uv add ../zero-shot-classification

# Ou se quiser especificar o caminho completo
uv add c:\projetos\testes\zero-shot-classification
```

## 📋 Verificação da Instalação

Após a instalação, teste se tudo funcionou:

```python
# Teste básico de importação
import deeprl_neural
print(f"✅ deeprl-neural v{deeprl_neural.__version__} instalado com sucesso!")

# Teste de componentes principais
from deeprl_neural import GridWorld, QLearningAgent, NeuralNetwork

# Criar instâncias para testar
env = GridWorld(grid_size=3)
agent = QLearningAgent(state_size=9, action_size=4)
nn = NeuralNetwork([2, 4, 1])

print("✅ Todos os componentes funcionando!")
```

## 🎯 Exemplo de Uso Básico

Depois de instalar, você pode usar a biblioteca assim:

```python
from deeprl_neural import (
    GridWorld, QLearningAgent, NeuralNetwork, 
    TrainingLogger, Matrix
)

# 1. Criar ambiente
env = GridWorld(grid_size=4)

# 2. Criar agente Q-Learning
agent = QLearningAgent(
    state_size=16,  # 4x4 = 16 estados possíveis
    action_size=4,  # up, down, left, right
    learning_rate=0.1,
    discount_factor=0.9
)

# 3. Treinar o agente
logger = TrainingLogger()

for episode in range(100):
    env.reset()
    state_index = env.get_state_index()
    total_reward = 0
    done = False
    
    while not done:
        action = agent.select_action(state_index)
        next_state, reward, done, info = env.step(action)
        next_state_index = env.get_state_index(next_state)
        
        agent.update(state_index, action, reward, next_state_index, done)
        
        state_index = next_state_index
        total_reward += reward
    
    logger.log_episode(episode, total_reward, info.get('steps', 0), info.get('success', False))

# 4. Ver resultados
performance = logger.get_recent_performance(10)
print(f"Taxa de sucesso: {performance['success_rate']:.1%}")
```

## 🖥️ Comandos CLI

A biblioteca também instala comandos de linha de comando:

```bash
# Ver demonstração interativa
deeprl-demo

# Treinar um agente
deeprl-train --agent qlearning --episodes 1000

# Executar testes
deeprl-test --component all
```

## 📚 Componentes Disponíveis

### Neural Networks
```python
from deeprl_neural import NeuralNetwork, Matrix

# Criar rede neural
nn = NeuralNetwork([4, 8, 2])  # 4 entradas, 8 ocultos, 2 saídas

# Forward pass
input_data = Matrix([[1.0, 0.5, -0.3, 0.8]])
output = nn.forward(input_data)
```

### Reinforcement Learning
```python
from deeprl_neural import QLearningAgent, DQNAgent

# Q-Learning para estados discretos
q_agent = QLearningAgent(state_size=10, action_size=4)

# DQN para estados contínuos
network = NeuralNetwork([4, 32, 16, 4])
dqn_agent = DQNAgent(network=network, learning_rate=0.001)
```

### Ambientes
```python
from deeprl_neural import GridWorld

# Mundo em grade simples
env = GridWorld(grid_size=5)

# Ambiente personalizado
env = GridWorld(
    grid_size=4,
    start_pos=(0, 0),
    goal_pos=(3, 3),
    walls=[(1, 1), (2, 2)],
    penalties=[(1, 2), (2, 1)]
)
```

### Utilitários
```python
from deeprl_neural import TrainingLogger, TrainingVisualizer

# Logging de treinamento
logger = TrainingLogger()

# Visualização (modo texto)
visualizer = TrainingVisualizer()
```

## 🔧 Resolução de Problemas

### Erro de Importação
```
ModuleNotFoundError: No module named 'deeprl_neural'
```

**Solução:**
1. Verifique se a instalação foi feita corretamente
2. Confirme que está no ambiente virtual correto
3. Reinstale a biblioteca

### Erro de Dependências
A biblioteca é **Pure Python** e não tem dependências externas, então não deve haver problemas de dependências.

### Problemas de Versão do Python
A biblioteca requer **Python >= 3.8**. Verifique sua versão:

```bash
python --version
```

## 📄 Requisitos do Sistema

- **Python**: >= 3.8
- **Dependências**: Nenhuma (Pure Python)
- **Sistema Operacional**: Qualquer (Windows, Linux, macOS)

## 📖 Documentação Adicional

- `README_LIBRARY.md`: Documentação completa da biblioteca
- `CONTRIBUTING.md`: Guia para contribuições
- `CHANGELOG.md`: Histórico de mudanças

## 💡 Exemplo Completo de Projeto

```python
# meu_projeto_rl.py
from deeprl_neural import *

def main():
    print("🚀 Meu Projeto de RL com deeprl-neural")
    
    # Configurar ambiente
    env = GridWorld(grid_size=3)
    agent = QLearningAgent(9, 4, learning_rate=0.2)
    
    # Treinar
    for episode in range(50):
        env.reset()
        state = env.get_state_index()
        
        while True:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            next_state_idx = env.get_state_index(next_state)
            
            agent.update(state, action, reward, next_state_idx, done)
            
            if done:
                break
            state = next_state_idx
    
    print("✅ Treinamento concluído!")

if __name__ == "__main__":
    main()
```

---

**🎉 Pronto! Agora você pode usar a biblioteca `deeprl-neural` em qualquer projeto Python!**
