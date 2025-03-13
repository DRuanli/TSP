import numpy as np
import matplotlib.pyplot as plt
import torch
import os
import sys
import time
import argparse

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from environment.delivery_env import DeliveryEnvironment
from agents.dqn_agent import DQNAgent
from utils.utils import create_distance_matrix, create_traffic_conditions, visualize_route, plot_learning_curve
from utils.google_maps_api import create_distance_matrix_with_api

def train_dqn_agent(
    num_locations=5,
    num_episodes=1000,
    use_google_maps=False,
    save_path="trained_models/dqn_agent.pth",
    plot_interval=100,
    save_interval=100,
    render=False
):
    """
    Train a DQN agent for the delivery route optimization problem.
    
    Args:
        num_locations (int): Number of locations (including depot)
        num_episodes (int): Number of training episodes
        use_google_maps (bool): Whether to use Google Maps API for distance matrix
        save_path (str): Path to save the trained agent
        plot_interval (int): Interval to plot training progress
        save_interval (int): Interval to save the agent
        render (bool): Whether to render the environment during training
    """
    print("Setting up the Delivery Route Optimization with DQN")
    
    # Create sample locations (2D coordinates)
    np.random.seed(42)  # For reproducibility
    locations = [(0, 0)]  # Depot at origin
    
    # Generate random locations
    for _ in range(num_locations - 1):
        location = (np.random.uniform(-10, 10), np.random.uniform(-10, 10))
        locations.append(location)
    
    print(f"Generated {num_locations} locations (including depot)")
    
    # Create distance matrix
    if use_google_maps:
        print("Using Google Maps API to create distance matrix")
        distance_matrix = create_distance_matrix_with_api(locations)
    else:
        print("Using Euclidean distance to create distance matrix")
        distance_matrix = create_distance_matrix(locations)
        
    print("Distance matrix shape:", distance_matrix.shape)
    
    # Create traffic conditions
    traffic_conditions = create_traffic_conditions(distance_matrix)
    print("Traffic conditions matrix shape:", traffic_conditions.shape)
    
    # Initialize environment
    env = DeliveryEnvironment(locations, distance_matrix, traffic_conditions)
    print("Environment initialized")
    
    # Initialize DQN agent
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dims=[128, 128],
        learning_rate=0.001,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        buffer_size=10000,
        batch_size=64,
        target_update_freq=10
    )
    print("DQN Agent initialized")
    
    # Training loop
    episode_rewards = []
    episode_losses = []
    episode_steps = []
    best_reward = float('-inf')
    
    for episode in range(1, num_episodes + 1):
        start_time = time.time()
        
        # Reset environment
        state = env.reset()
        done = False
        total_reward = 0
        losses = []
        steps = 0
        route = [0]  # Start at depot
        
        while not done:
            # Get valid actions
            valid_actions = env.get_valid_actions()
            
            # Select action
            action = agent.select_action(state, valid_actions)
            
            # Take action
            next_state, reward, done, info = env.step(action)
            
            # Store in replay buffer
            agent.store_transition(state, action, reward, next_state, done)
            
            # Update agent
            loss = agent.update()
            if loss > 0:
                losses.append(loss)
            
            # Update state and total reward
            state = next_state
            total_reward += reward
            steps += 1
            
            # Record route
            if info["valid_action"]:
                route.append(action)
            
            # Render environment
            if render:
                env.render()
                time.sleep(0.1)  # Add delay for visualization
        
        # Record episode statistics
        episode_rewards.append(total_reward)
        if losses:
            episode_losses.append(np.mean(losses))
        else:
            episode_losses.append(0)
        episode_steps.append(steps)
        
        # Save the best model
        if total_reward > best_reward:
            best_reward = total_reward
            agent.save(save_path)
        
        # Print progress
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        if episode % 10 == 0:
            print(f"Episode {episode}/{num_episodes} | Reward: {total_reward:.2f} | "
                 f"Avg Loss: {np.mean(losses) if losses else 0:.4f} | "
                 f"Steps: {steps} | Epsilon: {agent.epsilon:.4f} | "
                 f"Time: {elapsed_time:.2f}s")
        
        # Plot progress
        if episode % plot_interval == 0 or episode == num_episodes:
            # Plot learning curve
            plt = plot_learning_curve(episode_rewards)
            plt.savefig(f"learning_curve_episode_{episode}.png")
            plt.close()
            
            # Visualize final route
            plt = visualize_route(locations, route, f"Route at Episode {episode}")
            plt.savefig(f"route_episode_{episode}.png")
            plt.close()
        
        # Save checkpoint
        if episode % save_interval == 0:
            agent.save(f"checkpoint_episode_{episode}.pth")
    
    print("\nTraining completed!")
    print(f"Best reward: {best_reward:.2f}")
    print(f"Agent saved to {save_path}")
    
    # Return the trained agent and training history
    return agent, {
        "rewards": episode_rewards,
        "losses": episode_losses,
        "steps": episode_steps
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DQN agent for delivery route optimization")
    parser.add_argument("--num_locations", type=int, default=5, help="Number of locations including depot")
    parser.add_argument("--num_episodes", type=int, default=1000, help="Number of training episodes")
    parser.add_argument("--use_google_maps", action="store_true", help="Use Google Maps API for distance matrix")
    parser.add_argument("--save_path", type=str, default="trained_models/dqn_agent.pth", help="Path to save the trained agent")
    parser.add_argument("--plot_interval", type=int, default=100, help="Interval to plot training progress")
    parser.add_argument("--save_interval", type=int, default=100, help="Interval to save the agent")
    parser.add_argument("--render", action="store_true", help="Render the environment during training")
    
    args = parser.parse_args()
    
    # Create directory for trained models if it doesn't exist
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    
    # Train the agent
    train_dqn_agent(
        num_locations=args.num_locations,
        num_episodes=args.num_episodes,
        use_google_maps=args.use_google_maps,
        save_path=args.save_path,
        plot_interval=args.plot_interval,
        save_interval=args.save_interval,
        render=args.render
    )