# Rede Neural Deep Learning com Aprendizado por Reforço

## 🎯 Objetivo

Implementar uma rede neural deep learning com aprendizado por reforço **sem usar bibliotecas externas**, utilizando apenas Python puro e bibliotecas padrão.

## 🛠️ Características Principais

- **Zero Dependências**: Implementação puramente em Python sem numpy, tensorflow, pytorch, etc.
- **Aprendizado por Reforço**: Algoritmo Q-Learning (Deep Q-Network)
- **Persistência**: Sistema completo de salvamento e carregamento de modelos
- **Modular**: Arquitetura bem estruturada e extensível

## 📋 Status do Projeto

Consulte o arquivo `CHECKLIST_IMPLEMENTACAO.md` para acompanhar o progresso detalhado da implementação.

## 🏗️ Arquitetura

### Componentes Principais

1. **Neural Network Engine** (`src/neural_network/`)
   - Operações matemáticas básicas (álgebra linear)
   - Neurônios, camadas e rede neural completa
   - Funções de ativação e suas derivadas
   - Algoritmo de backpropagation

2. **Reinforcement Learning** (`src/reinforcement_learning/`)
   - Agente Q-learning
   - Experience replay buffer
   - Políticas de exploração (epsilon-greedy)
   - Target network para estabilidade

3. **Persistence System** (`src/persistence/`)
   - Serialização/deserialização de modelos
   - Salvamento de pesos e arquitetura
   - Sistema de versionamento
   - Validação de integridade

4. **Environment Framework** (`src/environment/`)
   - Interface base para ambientes
   - Ambientes de teste (GridWorld, CartPole simulado)
   - Sistema de recompensas

## 🚀 Como Usar

### Treinamento Básico
```python
from src.reinforcement_learning.agent import QLearningAgent
from src.environment.test_envs import GridWorld

# Criar ambiente
env = GridWorld(size=5)

# Criar agente
agent = QLearningAgent(
    state_size=env.state_size,
    action_size=env.action_size,
    learning_rate=0.001
)

# Treinar
for episode in range(1000):
    state = env.reset()
    total_reward = 0
    
    while not env.done:
        action = agent.choose_action(state)
        next_state, reward, done = env.step(action)
        agent.remember(state, action, reward, next_state, done)
        agent.replay()
        state = next_state
        total_reward += reward
    
    print(f"Episode {episode}: Reward = {total_reward}")
```

### Salvamento e Carregamento
```python
from src.persistence.model_saver import ModelSaver
from src.persistence.model_loader import ModelLoader

# Salvar modelo
saver = ModelSaver()
saver.save(agent, "models/trained_agent.json")

# Carregar modelo
loader = ModelLoader()
loaded_agent = loader.load("models/trained_agent.json")
```

## 📦 Dependências

**Nenhuma dependência externa!** Este projeto usa apenas:
- Python 3.7+ (bibliotecas padrão)
- `json` (serialização)
- `random` (geração de números aleatórios)
- `math` (funções matemáticas)
- `pickle` (serialização alternativa)

## 📊 Algoritmos Implementados

### Deep Q-Network (DQN)
- Q-learning com aproximação de função neural
- Experience replay para estabilidade
- Target network para reduzir correlação
- Epsilon-greedy para exploração

### Componentes Matemáticos
- Forward propagation
- Backpropagation
- Gradient descent
- Funções de ativação (Sigmoid, ReLU, Tanh, Softmax)

## 📈 Performance Esperada

- **Convergência**: 500-2000 episódios (dependendo do ambiente)
- **Memória**: ~10-50MB (dependendo do tamanho da rede)
- **Velocidade**: 100-1000 passos/segundo (CPU)

## 🧪 Testes

Execute os testes para validar a implementação:

```bash
python -m tests.test_neural_network
python -m tests.test_rl_agent
python -m tests.test_persistence
python -m tests.test_environments
```

## 📚 Estrutura de Aprendizado

### Ambientes Incluídos
1. **GridWorld**: Ambiente 2D simples para testes básicos
2. **CartPole**: Simulação do problema clássico de controle
3. **Custom Environment**: Framework para criar ambientes personalizados

### Métricas de Acompanhamento
- Recompensa média por episódio
- Taxa de convergência
- Loss da rede neural
- Epsilon decay progression

## 🔧 Configuração

Edite `src/config.py` para ajustar hiperparâmetros:

```python
# Arquitetura da rede
HIDDEN_LAYERS = [64, 64]
ACTIVATION_FUNCTION = "relu"

# Hiperparâmetros de RL
LEARNING_RATE = 0.001
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995

# Experience Replay
BUFFER_SIZE = 10000
BATCH_SIZE = 32
```

## 📖 Exemplos

Veja a pasta `examples/` para:
- `simple_training.py`: Treinamento básico
- `save_load_demo.py`: Demonstração de persistência
- `custom_environment.py`: Como criar ambientes personalizados

## 🤝 Contribuição

1. Siga o checklist de implementação
2. Mantenha o código sem dependências externas
3. Adicione testes para novas funcionalidades
4. Documente mudanças significativas

## 📄 Licença

MIT License - Veja arquivo LICENSE para detalhes.

---

**Início do Projeto**: Agosto 2025  
**Status**: 🟡 Em Planejamento  
**Estimativa**: 8 semanas para conclusão completa
