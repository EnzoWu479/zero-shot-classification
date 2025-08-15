"""
Example demonstrations for DeepRL Neural Library
"""

def run_demo():
    """Run a demonstration of the library capabilities."""
    print("🚀 DeepRL Neural Library Demo")
    print("=" * 50)
    
    try:
        from deeprl_neural import (
            GridWorld, QLearningAgent, DQNAgent, NeuralNetwork,
            TrainingLogger, TrainingVisualizer
        )
        
        print("\n1. Creating Grid World Environment...")
        env = GridWorld(grid_size=4)
        print(f"   ✅ Environment created (size: {env.grid_size}x{env.grid_size})")
        print(f"   📊 State space: {env.grid_size * env.grid_size}")
        print(f"   🎮 Action space: {env.get_action_size()}")
        
        print("\n2. Training Q-Learning Agent...")
        qlearning_agent = QLearningAgent(
            state_size=env.grid_size * env.grid_size,  # Total number of states
            action_size=env.get_action_size(),
            learning_rate=0.1,
            discount_factor=0.9
        )
        
        # Quick training
        logger = TrainingLogger()
        for episode in range(50):
            env.reset()
            state_index = env.get_state_index()  # Get state as index
            total_reward = 0
            done = False
            
            while not done:
                action = qlearning_agent.select_action(state_index)
                next_state, reward, done, info = env.step(action)
                next_state_index = env.get_state_index(next_state)  # Convert to index
                qlearning_agent.update(state_index, action, reward, next_state_index, done)
                state_index = next_state_index
                total_reward += reward
            
            logger.log_episode(episode, total_reward, info.get('steps', 0), info.get('success', False))
        
        q_perf = logger.get_recent_performance(50)
        print(f"   ✅ Q-Learning trained: {q_perf['success_rate']:.1%} success rate")
        
        print("\n3. Training DQN Agent...")
        network = NeuralNetwork([env.get_state_size(), 32, 16, env.get_action_size()])
        dqn_agent = DQNAgent(network=network, learning_rate=0.001)
        
        logger = TrainingLogger()
        for episode in range(50):
            state = env.reset()
            total_reward = 0
            done = False
            
            while not done:
                action = dqn_agent.select_action(state)
                next_state, reward, done, info = env.step(action)
                dqn_agent.update(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
            
            logger.log_episode(episode, total_reward, info.get('steps', 0), info.get('success', False))
        
        dqn_perf = logger.get_recent_performance(50)
        print(f"   ✅ DQN trained: {dqn_perf['success_rate']:.1%} success rate")
        
        print("\n4. Neural Network Demo...")
        # Create a simple neural network
        nn = NeuralNetwork([2, 4, 1], ['relu', 'sigmoid'])
        
        # Test forward pass
        from deeprl_neural import Matrix
        input_data = Matrix([[0.5, 0.3]])
        output = nn.forward(input_data)
        print(f"   ✅ Neural Network forward pass: input {input_data.data[0]} → output {output.data[0][0]:.4f}")
        
        print("\n5. Visualization Demo...")
        viz = TrainingVisualizer()
        
        # Create some sample data
        rewards = [q_perf['avg_reward'], dqn_perf['avg_reward']]
        success_rates = [q_perf['success_rate'], dqn_perf['success_rate']]
        
        print("   📊 Agent Performance Comparison:")
        print(f"   Q-Learning: Reward={rewards[0]:.2f}, Success={success_rates[0]:.1%}")
        print(f"   DQN:        Reward={rewards[1]:.2f}, Success={success_rates[1]:.1%}")
        
        print("\n6. Library Information...")
        from deeprl_neural import get_info
        info = get_info()
        print(f"   📚 Library: {info['name']} v{info['version']}")
        print(f"   👥 Author: {info['author']}")
        print(f"   📝 License: {info['license']}")
        print("\n   🧩 Available Components:")
        for component, description in info['components'].items():
            print(f"     • {component}: {description}")
        
        print("\n🎉 Demo completed successfully!")
        print("\nNext steps:")
        print("  • Try: 'deeprl-train --agent qlearning --episodes 1000'")
        print("  • Try: 'deeprl-test --component all'")
        print("  • Import the library: 'from deeprl_neural import *'")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure the library is properly installed.")
    except Exception as e:
        print(f"❌ Demo error: {e}")


def neural_network_example():
    """Demonstrate neural network capabilities."""
    from deeprl_neural import NeuralNetwork, Matrix, sigmoid, relu
    
    print("Neural Network Example")
    print("-" * 30)
    
    # Create network
    network = NeuralNetwork([3, 5, 2], ['relu', 'sigmoid'])
    print(f"Created network: {network.layer_sizes}")
    
    # Forward pass
    input_data = Matrix([[1.0, 0.5, -0.3]])
    output = network.forward(input_data)
    print(f"Forward pass: {input_data.data[0]} → {output.data[0]}")
    
    # Training example
    target = Matrix([[1.0, 0.0]])
    loss = network.train(input_data, target, learning_rate=0.01)
    print(f"Training loss: {loss}")


def reinforcement_learning_example():
    """Demonstrate reinforcement learning capabilities."""
    from deeprl_neural import GridWorld, QLearningAgent
    
    print("Reinforcement Learning Example")
    print("-" * 30)
    
    # Create environment and agent
    env = GridWorld(grid_size=3)
    agent = QLearningAgent(
        state_size=env.get_state_size(),
        action_size=env.get_action_size()
    )
    
    print(f"Environment: {env.grid_size}x{env.grid_size} grid")
    print(f"Agent: Q-Learning")
    
    # Run a few episodes
    for episode in range(5):
        state = env.reset()
        total_reward = 0
        steps = 0
        
        while steps < 20:  # Max steps to avoid infinite loop
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            agent.update(state, action, reward, next_state, done)
            
            state = next_state
            total_reward += reward
            steps += 1
            
            if done:
                break
        
        print(f"Episode {episode + 1}: {steps} steps, reward {total_reward:.2f}")


if __name__ == "__main__":
    run_demo()
