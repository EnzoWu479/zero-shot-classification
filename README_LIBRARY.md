# DeepRL Neural 🧠🤖

[![PyPI version](https://badge.fury.io/py/deeprl-neural.svg)](https://badge.fury.io/py/deeprl-neural)
[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://github.com/deeprl-team/deeprl-neural/workflows/Tests/badge.svg)](https://github.com/deeprl-team/deeprl-neural/actions)
[![Coverage](https://codecov.io/gh/deeprl-team/deeprl-neural/branch/main/graph/badge.svg)](https://codecov.io/gh/deeprl-team/deeprl-neural)

**Pure Python Deep Reinforcement Learning Library with Neural Networks**

A comprehensive, educational implementation of deep reinforcement learning algorithms with zero external dependencies. Perfect for learning, teaching, and research.

## ✨ Features

- 🚀 **Pure Python**: No external dependencies required
- 🧠 **Complete Neural Networks**: Full backpropagation implementation
- 🎮 **RL Algorithms**: Q-Learning and Deep Q-Networks (DQN)
- 🌍 **Grid World Environment**: Built-in testing environment
- 💾 **Model Persistence**: Save and load trained models
- 📊 **Visualization Tools**: Training progress and performance metrics
- 🧪 **Comprehensive Tests**: 67+ test coverage with pytest
- 📚 **Educational**: Well-documented and easy to understand

## 🚀 Quick Start

### Installation

```bash
pip install deeprl-neural
```

### Basic Usage

```python
from deeprl_neural import GridWorld, QLearningAgent, NeuralNetwork

# Create environment
env = GridWorld(size=5)

# Create and train Q-Learning agent
agent = QLearningAgent(
    state_size=env.get_state_size(),
    action_size=env.get_action_size()
)

# Training loop
for episode in range(1000):
    state = env.reset()
    done = False
    
    while not done:
        action = agent.choose_action(state)
        next_state, reward, done, info = env.step(action)
        agent.learn(state, action, reward, next_state, done)
        state = next_state

print(f"Training completed! Success rate: {info.get('success_rate', 0):.2%}")
```

### Neural Network Example

```python
from deeprl_neural import NeuralNetwork, Matrix

# Create neural network
network = NeuralNetwork([4, 16, 8, 2], ['relu', 'relu', 'sigmoid'])

# Forward pass
input_data = Matrix([[1.0, 0.5, -0.3, 0.8]])
output = network.forward(input_data)

# Training
target = Matrix([[1.0, 0.0]])
loss = network.train(input_data, target, learning_rate=0.01)
```

### DQN Agent Example

```python
from deeprl_neural import DQNAgent, NeuralNetwork, GridWorld

# Create environment and neural network
env = GridWorld(size=4)
network = NeuralNetwork([env.get_state_size(), 64, 32, env.get_action_size()])

# Create DQN agent
agent = DQNAgent(network=network, learning_rate=0.001)

# Train the agent
for episode in range(500):
    state = env.reset()
    done = False
    
    while not done:
        action = agent.choose_action(state)
        next_state, reward, done, info = env.step(action)
        agent.learn(state, action, reward, next_state, done)
        state = next_state
```

## 🛠️ Command Line Interface

DeepRL Neural includes convenient CLI tools:

```bash
# Train an agent
deeprl-train --agent qlearning --episodes 1000 --save my_model.json

# Train a DQN agent
deeprl-train --agent dqn --episodes 500 --size 6 --verbose

# Run tests
deeprl-test --component all

# Quick demo
deeprl-demo
```

## 📦 Library Components

### Neural Networks
- **Matrix Operations**: Pure Python matrix implementation
- **Activation Functions**: Sigmoid, ReLU, Tanh, Linear, Softmax
- **Network Architecture**: Flexible layer configuration
- **Backpropagation**: Complete training implementation

### Reinforcement Learning
- **Q-Learning**: Classic tabular RL algorithm
- **Deep Q-Networks (DQN)**: Neural network-based RL
- **Experience Replay**: Training stability improvements
- **Exploration Policies**: ε-greedy, Boltzmann, UCB

### Environment
- **Grid World**: Configurable testing environment
- **Multiple Layouts**: Simple, maze, and random configurations
- **Customizable**: Rewards, obstacles, and goals

### Utilities
- **Training Logger**: Track performance metrics
- **Early Stopping**: Prevent overfitting
- **Visualization**: ASCII charts and progress tracking
- **Model Persistence**: Save/load functionality

## 📊 Performance

The library achieves excellent performance on standard benchmarks:

| Algorithm | Environment | Success Rate | Training Episodes |
|-----------|-------------|--------------|-------------------|
| Q-Learning | GridWorld 5x5 | 95%+ | 500-1000 |
| DQN | GridWorld 5x5 | 90%+ | 800-1500 |
| Q-Learning | GridWorld 10x10 | 85%+ | 1500-3000 |

## 🔧 Advanced Usage

### Custom Neural Network

```python
from deeprl_neural import NeuralNetwork, sigmoid, relu

# Custom architecture
network = NeuralNetwork(
    layer_sizes=[10, 20, 15, 5],
    activations=[relu, sigmoid, relu]
)

# Custom training
for epoch in range(100):
    for batch in training_data:
        loss = network.train(batch.inputs, batch.targets)
```

### Model Persistence

```python
from deeprl_neural import save_model, load_model

# Save trained agent
save_model(agent, "trained_agent.json")

# Load agent later
loaded_agent = load_model("trained_agent.json")
```

### Custom Environment

```python
from deeprl_neural import GridWorld

# Custom grid world
env = GridWorld(
    size=8,
    obstacles=[(2, 3), (4, 5)],
    goal=(7, 7),
    penalty_cells=[(1, 1), (2, 2)]
)
```

## 🧪 Testing

The library includes comprehensive tests:

```bash
# Run all tests
pytest tests/

# Run specific test categories
pytest tests/ -m "unit"
pytest tests/ -m "integration"
pytest tests/ -m "neural"

# Generate coverage report
pytest tests/ --cov=deeprl_neural --cov-report=html
```

Current test coverage: **67%** with **108+ passing tests**

## 📚 Documentation

- [API Reference](https://deeprl-neural.readthedocs.io/api/)
- [Tutorial Notebooks](https://github.com/deeprl-team/deeprl-neural/tree/main/examples)
- [Algorithm Explanations](https://deeprl-neural.readthedocs.io/theory/)
- [Contributing Guide](https://github.com/deeprl-team/deeprl-neural/blob/main/CONTRIBUTING.md)

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

```bash
git clone https://github.com/deeprl-team/deeprl-neural.git
cd deeprl-neural
pip install -e .[dev]
pre-commit install
```

### Running Tests

```bash
pytest tests/
black deeprl_neural/
flake8 deeprl_neural/
mypy deeprl_neural/
```

## 📈 Roadmap

- [ ] Additional RL algorithms (A3C, PPO, DDPG)
- [ ] More environments (CartPole, Mountain Car)
- [ ] Convolutional neural networks
- [ ] GPU acceleration support
- [ ] Interactive Jupyter notebooks
- [ ] Web-based visualization dashboard

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Inspired by classic RL textbooks and papers
- Built for educational purposes and research
- Special thanks to the Python community

## 📞 Support

- 📧 Email: team@deeprl.dev
- 🐛 Issues: [GitHub Issues](https://github.com/deeprl-team/deeprl-neural/issues)
- 💬 Discussions: [GitHub Discussions](https://github.com/deeprl-team/deeprl-neural/discussions)
- 📖 Documentation: [Read the Docs](https://deeprl-neural.readthedocs.io)

---

Made with ❤️ by the DeepRL Team
