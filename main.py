import numpy as np
import matplotlib.pyplot as plt
import torch
import os
import sys

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from environment.delivery_env import DeliveryEnvironment
from agents.dqn_agent import DQNAgent
from utils.utils import create_distance_matrix, create_traffic_conditions, visualize_route


def main():
    """
    Main function to demonstrate Task 1 setup for the DQN delivery route optimization.
    """
    print("Setting up the Delivery Route Optimization with DQN")
    
    # Create sample locations (2D coordinates)
    num_locations = 5  # Including depot
    np.random.seed(42)  # For reproducibility
    locations = [(0, 0)]  # Depot at origin
    
    # Generate random locations
    for _ in range(num_locations - 1):
        location = (np.random.uniform(-10, 10), np.random.uniform(-10, 10))
        locations.append(location)
    
    print(f"Generated {num_locations} locations (including depot)")
    
    # Create distance matrix
    distance_matrix = create_distance_matrix(locations)
    print("Distance matrix shape:", distance_matrix.shape)
    
    # Create traffic conditions
    traffic_conditions = create_traffic_conditions(distance_matrix)
    print("Traffic conditions matrix shape:", traffic_conditions.shape)
    
    # Initialize environment
    env = DeliveryEnvironment(locations, distance_matrix, traffic_conditions)
    print("Environment initialized")
    
    # Print environment details
    print("\nEnvironment Details:")
    print(f"State shape: {env.observation_space.shape}")
    print(f"Action space: {env.action_space}")
    
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
    print("\nDQN Agent initialized")
    
    # Print agent details
    print("\nAgent Details:")
    print(f"State dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")
    print(f"Hidden layers: {[128, 128]}")
    print(f"Learning rate: {0.001}")
    print(f"Discount factor (gamma): {0.99}")
    print(f"Initial epsilon: {1.0}")
    print(f"Replay buffer size: {10000}")
    
    # Demonstrate a single episode
    print("\nDemonstrating a single episode with random actions:")
    state = env.reset()
    done = False
    total_reward = 0
    route = [0]  # Start at depot
    
    while not done:
        # Get valid actions
        valid_actions = env.get_valid_actions()
        
        # Select action (random for demonstration)
        action = agent.select_action(state, valid_actions)
        
        # Take action
        next_state, reward, done, info = env.step(action)
        
        # Store in replay buffer
        agent.store_transition(state, action, reward, next_state, done)
        
        # Update state and total reward
        state = next_state
        total_reward += reward
        
        # Record route
        if info["valid_action"]:
            route.append(action)
        
        # Render environment
        env.render()
        print("---")
    
    print(f"\nEpisode finished with total reward: {total_reward:.2f}")
    print(f"Route taken: {route}")
    
    # Visualize the route
    plt = visualize_route(locations, route, "Random Policy Route")
    plt.savefig("random_route.png")
    print("\nRoute visualization saved to 'random_route.png'")
    
    print("\nTask 1 completed successfully!")


if __name__ == "__main__":
    main()