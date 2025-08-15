# 🚀 Como Instalar deeprl-neural em Outros Projetos

## 📋 Resumo Rápido

A biblioteca `deeprl-neural` é uma implementação **Pure Python** de Deep Reinforcement Learning com zero dependências externas. Aqui estão as formas de instalá-la:

## 🎯 Método 1: Instalação via Wheel (RECOMENDADO)

```bash
# 1. Na pasta da biblioteca, gere o wheel (já feito!)
cd c:\projetos\testes\zero-shot-classification
uv run python -m build

# 2. No seu projeto, instale o wheel
pip install c:\projetos\testes\zero-shot-classification\dist\deeprl_neural-1.0.0-py3-none-any.whl
```

## 🔧 Método 2: Instalação em Modo Desenvolvimento

```bash
# Para desenvolvimento ativo da biblioteca
pip install -e c:\projetos\testes\zero-shot-classification
```

## 📦 Método 3: Com UV (Gerenciador Moderno)

```bash
cd meu-projeto
uv add c:\projetos\testes\zero-shot-classification
```

## ✅ Verificação da Instalação

```python
import deeprl_neural
print(f"✅ deeprl-neural v{deeprl_neural.__version__} instalado!")

# Teste rápido
from deeprl_neural import GridWorld, QLearningAgent, NeuralNetwork
env = GridWorld(grid_size=3)
agent = QLearningAgent(9, 4)
nn = NeuralNetwork([2, 4, 1])
print("✅ Todos os componentes funcionando!")
```

## 🎮 Uso Básico

```python
from deeprl_neural import *

# Criar ambiente
env = GridWorld(grid_size=4)

# Criar agente Q-Learning  
agent = QLearningAgent(
    state_size=16,  # 4x4 = 16 estados
    action_size=4,  # up, down, left, right
    learning_rate=0.1,
    discount_factor=0.9
)

# Treinar
for episode in range(100):
    env.reset()
    state = env.get_state_index()
    
    while True:
        action = agent.select_action(state)
        next_state, reward, done, info = env.step(action)
        next_state_idx = env.get_state_index(next_state)
        
        agent.update(state, action, reward, next_state_idx, done)
        
        if done:
            break
        state = next_state_idx
```

## 🖥️ Comandos CLI

Após instalar, você tem acesso aos comandos:

```bash
deeprl-demo         # Demonstração interativa
deeprl-train        # Treinamento de agentes  
deeprl-test         # Execução de testes
```

## 📚 Componentes Disponíveis

- **GridWorld**: Ambiente de teste
- **QLearningAgent**: Agente Q-Learning clássico
- **DQNAgent**: Deep Q-Network
- **NeuralNetwork**: Rede neural do zero
- **TrainingLogger**: Log de treinamento
- **Matrix**: Operações matriciais
- **ModelSaver**: Salvar/carregar modelos

## 🎯 Exemplo de Projeto Completo

```python
# meu_projeto.py
from deeprl_neural import *

def main():
    print("🚀 Meu Projeto RL")
    
    # Setup
    env = GridWorld(grid_size=4)
    agent = QLearningAgent(16, 4, learning_rate=0.2)
    logger = TrainingLogger()
    
    # Treinar 200 episódios
    for episode in range(200):
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
        
        if (episode + 1) % 50 == 0:
            perf = logger.get_recent_performance(50)
            print(f"Episódio {episode + 1}: {perf['success_rate']:.1%} sucesso")
    
    print("✅ Treinamento concluído!")

if __name__ == "__main__":
    main()
```

## 📋 Requisitos

- **Python**: >= 3.8
- **Dependências**: Nenhuma! (Pure Python)
- **Sistema**: Windows, Linux, macOS

## 📁 Arquivos Gerados

Após `uv run python -m build`:
- `dist/deeprl_neural-1.0.0-py3-none-any.whl` ← Use este!
- `dist/deeprl_neural-1.0.0.tar.gz` ← Source distribution

## 🔨 Script de Instalação Automática

Use o script incluído:

```bash
python install_script.py wheel    # Gera e instala wheel
python install_script.py dev      # Instala modo desenvolvimento  
python install_script.py test     # Testa instalação
```

---

**🎉 Pronto! Sua biblioteca deeprl-neural está pronta para ser usada em qualquer projeto Python!**

## 📖 Documentação Adicional

- `INSTALLATION_GUIDE.md` - Guia detalhado de instalação
- `PROJECT_EXAMPLE.md` - Exemplo completo de projeto
- `README_LIBRARY.md` - Documentação da biblioteca
- `CONTRIBUTING.md` - Guia para contribuições
