"""
DeepRL Neural - Pure Python Deep Reinforcement Learning Library

A comprehensive, zero-dependency implementation of deep reinforcement learning
algorithms with neural networks, designed for education and research.

Key Features:
- Pure Python implementation (no external dependencies)
- Complete neural network framework with backpropagation
- Q-Learning and Deep Q-Network (DQN) agents
- Grid World environment for testing
- Model persistence and visualization tools
- Comprehensive test coverage

Example:
    >>> from deeprl_neural import NeuralNetwork, QLearningAgent, GridWorld
    >>> 
    >>> # Create environment
    >>> env = GridWorld(size=5)
    >>> 
    >>> # Create agent
    >>> agent = QLearningAgent(
    ...     state_size=env.get_state_size(),
    ...     action_size=env.get_action_size()
    ... )
    >>> 
    >>> # Train agent
    >>> for episode in range(100):
    ...     state = env.reset()
    ...     done = False
    ...     while not done:
    ...         action = agent.choose_action(state)
    ...         next_state, reward, done, info = env.step(action)
    ...         agent.learn(state, action, reward, next_state, done)
    ...         state = next_state

Authors: Deep RL Team
License: MIT
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "Deep RL Team"
__email__ = "team@deeprl.dev"
__license__ = "MIT"

# Core neural network components
from .neural_network.matrix import Matrix, zeros, ones, identity, random_matrix, vector
from .neural_network.activation import sigmoid, relu, tanh, linear, softmax
from .neural_network.neuron import Neuron
from .neural_network.layer import Layer
from .neural_network.network import NeuralNetwork

# Reinforcement learning components
from .reinforcement_learning.agents import QLearningAgent, DQNAgent, AgentFactory
from .reinforcement_learning.policies import (
    EpsilonGreedyPolicy, 
    BoltzmannPolicy, 
    UCBPolicy,
    PolicyFactory
)
from .reinforcement_learning.experience_replay import (
    ExperienceReplayBuffer,
    PrioritizedExperienceReplayBuffer
)

# Environment
from .environment.grid_world import GridWorld

# Utilities
from .utils.training_utils import (
    TrainingLogger,
    EarlyStopping,
    LearningRateScheduler,
    RewardShaper,
    ExperienceBuffer
)
from .utils.visualization import SimpleChart, TrainingVisualizer

# Persistence
from .persistence.model_saver import ModelSaver, save_model
from .persistence.model_loader import ModelLoader, load_model

# High-level API
__all__ = [
    # Version info
    "__version__", "__author__", "__email__", "__license__",
    
    # Neural Network
    "Matrix", "zeros", "ones", "identity", "random_matrix", "vector",
    "sigmoid", "relu", "tanh", "linear", "softmax",
    "Neuron", "Layer", "NeuralNetwork",
    
    # Reinforcement Learning
    "QLearningAgent", "DQNAgent", "AgentFactory",
    "EpsilonGreedyPolicy", "BoltzmannPolicy", "UCBPolicy", "PolicyFactory",
    "ExperienceReplayBuffer", "PrioritizedExperienceReplayBuffer",
    
    # Environment
    "GridWorld",
    
    # Utilities
    "TrainingLogger", "EarlyStopping", "LearningRateScheduler", 
    "RewardShaper", "ExperienceBuffer", "SimpleChart", "TrainingVisualizer",
    
    # Persistence
    "ModelSaver", "save_model", "ModelLoader", "load_model",
]

def get_version():
    """Get the library version."""
    return __version__

def get_info():
    """Get library information."""
    return {
        "name": "deeprl-neural",
        "version": __version__,
        "author": __author__,
        "email": __email__,
        "license": __license__,
        "description": "Pure Python Deep Reinforcement Learning Library",
        "components": {
            "neural_network": "Complete neural network implementation",
            "reinforcement_learning": "Q-Learning and DQN agents",
            "environment": "Grid World testing environment",
            "utils": "Training and visualization utilities",
            "persistence": "Model save/load functionality"
        }
    }
