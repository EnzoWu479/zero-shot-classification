"""
Command Line Interface for DeepRL Neural Library
"""

import argparse
import sys
import json
from typing import Dict, Any

def train_command():
    """Command line interface for training models."""
    parser = argparse.ArgumentParser(
        description='Train DeepRL Neural models',
        prog='deeprl-train'
    )
    
    parser.add_argument(
        '--agent', 
        choices=['qlearning', 'dqn'], 
        default='qlearning',
        help='Type of agent to train'
    )
    
    parser.add_argument(
        '--episodes', 
        type=int, 
        default=1000,
        help='Number of training episodes'
    )
    
    parser.add_argument(
        '--environment', 
        default='gridworld',
        help='Environment to use for training'
    )
    
    parser.add_argument(
        '--size', 
        type=int, 
        default=5,
        help='Size of the grid world environment'
    )
    
    parser.add_argument(
        '--save', 
        type=str,
        help='Path to save the trained model'
    )
    
    parser.add_argument(
        '--config', 
        type=str,
        help='JSON config file with training parameters'
    )
    
    parser.add_argument(
        '--verbose', 
        action='store_true',
        help='Enable verbose output'
    )
    
    args = parser.parse_args()
    
    try:
        from deeprl_neural import GridWorld, QLearningAgent, DQNAgent, NeuralNetwork
        from deeprl_neural.utils.training_utils import TrainingLogger
        from deeprl_neural.persistence import save_model
        
        # Load config if provided
        config = {}
        if args.config:
            with open(args.config, 'r') as f:
                config = json.load(f)
        
        # Create environment
        env = GridWorld(size=args.size)
        
        # Create agent
        if args.agent == 'qlearning':
            agent = QLearningAgent(
                state_size=env.get_state_size(),
                action_size=env.get_action_size(),
                **config.get('agent_params', {})
            )
        else:  # DQN
            network = NeuralNetwork([env.get_state_size(), 64, 32, env.get_action_size()])
            agent = DQNAgent(
                network=network,
                **config.get('agent_params', {})
            )
        
        # Training loop
        logger = TrainingLogger()
        
        if args.verbose:
            print(f"Training {args.agent} agent for {args.episodes} episodes...")
            print(f"Environment: {args.environment} (size: {args.size})")
        
        for episode in range(args.episodes):
            state = env.reset()
            total_reward = 0
            steps = 0
            done = False
            
            while not done:
                action = agent.choose_action(state)
                next_state, reward, done, info = env.step(action)
                agent.learn(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
                steps += 1
            
            logger.log_episode(episode, total_reward, steps, info.get('success', False))
            
            if args.verbose and (episode + 1) % 100 == 0:
                perf = logger.get_recent_performance(100)
                print(f"Episode {episode + 1}: Avg Reward = {perf['avg_reward']:.2f}, "
                      f"Success Rate = {perf['success_rate']:.2%}")
        
        # Save model if requested
        if args.save:
            save_model(agent, args.save)
            print(f"Model saved to: {args.save}")
        
        # Final statistics
        final_perf = logger.get_recent_performance(100)
        print(f"\nTraining completed!")
        print(f"Final Performance (last 100 episodes):")
        print(f"  Average Reward: {final_perf['avg_reward']:.2f}")
        print(f"  Success Rate: {final_perf['success_rate']:.2%}")
        print(f"  Average Steps: {final_perf['avg_steps']:.1f}")
        
    except Exception as e:
        print(f"Error during training: {e}")
        sys.exit(1)


def test_command():
    """Command line interface for testing the library."""
    parser = argparse.ArgumentParser(
        description='Test DeepRL Neural library',
        prog='deeprl-test'
    )
    
    parser.add_argument(
        '--component', 
        choices=['neural', 'rl', 'all'], 
        default='all',
        help='Component to test'
    )
    
    parser.add_argument(
        '--quick', 
        action='store_true',
        help='Run quick tests only'
    )
    
    args = parser.parse_args()
    
    try:
        import subprocess
        import os
        
        # Find the package directory
        import deeprl_neural
        package_dir = os.path.dirname(deeprl_neural.__file__)
        project_dir = os.path.dirname(package_dir)
        
        # Build pytest command
        cmd = ['python', '-m', 'pytest']
        
        if args.component == 'neural':
            cmd.extend(['-m', 'neural'])
        elif args.component == 'rl':
            cmd.extend(['-m', 'rl'])
        
        if args.quick:
            cmd.extend(['-m', 'not slow'])
        
        cmd.extend(['-v', os.path.join(project_dir, 'tests')])
        
        print(f"Running tests: {' '.join(cmd)}")
        result = subprocess.run(cmd, cwd=project_dir)
        sys.exit(result.returncode)
        
    except Exception as e:
        print(f"Error running tests: {e}")
        sys.exit(1)


if __name__ == "__main__":
    print("Use 'deeprl-train' or 'deeprl-test' commands")
