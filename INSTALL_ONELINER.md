# ⚡ INSTALAÇÃO RÁPIDA - deeprl-neural

## 🎯 Uma Linha de Comando

```bash
pip install c:\projetos\testes\zero-shot-classification\dist\deeprl_neural-1.0.0-py3-none-any.whl
```

## ✅ Verificar

```python
import deeprl_neural; print(f"v{deeprl_neural.__version__} OK!")
```

## 🚀 Usar

```python
from deeprl_neural import *

env = GridWorld(grid_size=4)
agent = QLearningAgent(16, 4)

for i in range(10):
    env.reset()
    state = env.get_state_index()
    action = agent.select_action(state)
    next_state, reward, done, info = env.step(action)
    agent.update(state, action, reward, env.get_state_index(next_state), done)
    print(f"Episódio {i+1}: Reward={reward}")
```

## 🖥️ CLI

```bash
deeprl-demo    # Ver demonstração
```

---

**🎉 Pronto! 3 comandos e você está usando RL avançado!**
