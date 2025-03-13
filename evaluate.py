import numpy as np
import matplotlib.pyplot as plt
import torch
import os
import sys
import argparse

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from environment.delivery_env import DeliveryEnvironment
from agents.dqn_agent import DQNAgent
from utils.utils import create_distance_matrix, create_traffic_conditions, visualize_route
from utils.google_maps_api import create_distance_matrix_with_api

def evaluate_agent(
    agent_path,
    num_locations=5,
    num_episodes=10,
    use_google_maps=False,
    render=True,
    seed=None
):
    """
    Evaluate a trained DQN agent on new sets of cities.
    
    Args:
        agent_path (str): Path to the trained agent
        num_locations (int): Number of locations (including depot)
        num_episodes (int): Number of evaluation episodes
        use_google_maps (bool): Whether to use Google Maps API for distance matrix
        render (bool): Whether to render the environment during evaluation
        seed (int): Random seed for reproducibility
    
    Returns:
        dict: Evaluation results
    """
    print(f"Evaluating agent from {agent_path}")
    
    # Set random seed if provided
    if seed is not None:
        np.random.seed(seed)
    
    # Initialize results
    results = {
        "rewards": [],
        "distances": [],
        "steps": [],
        "routes": []
    }
    
    for episode in range(1, num_episodes + 1):
        print(f"\nEvaluation Episode {episode}/{num_episodes}")
        
        # Create new set of locations
        locations = [(0, 0)]  # Depot at origin
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
        
        # Create traffic conditions
        traffic_conditions = create_traffic_conditions(distance_matrix)
        
        # Initialize environment
        env = DeliveryEnvironment(locations, distance_matrix, traffic_conditions)
        
        # Initialize agent
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n
        
        agent = DQNAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=[128, 128],
            learning_rate=0.001,
            gamma=0.99,
            epsilon_start=0.0,  # No exploration during evaluation
            epsilon_end=0.0,
            epsilon_decay=1.0,
            buffer_size=10000,
            batch_size=64,
            target_update_freq=10
        )
        
        # Load trained weights
        agent.load(agent_path)
        
        # Reset environment
        state = env.reset()
        done = False
        total_reward = 0
        steps = 0
        route = [0]  # Start at depot
        
        while not done:
            # Get valid actions
            valid_actions = env.get_valid_actions()
            
            # Select action (no exploration)
            action = agent.select_action(state, valid_actions)
            
            # Take action
            next_state, reward, done, info = env.step(action)
            
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
        
        # Record results
        results["rewards"].append(total_reward)
        results["distances"].append(env.total_distance)
        results["steps"].append(steps)
        results["routes"].append(route)
        
        print(f"Total reward: {total_reward:.2f}")
        print(f"Total distance: {env.total_distance:.2f}")
        print(f"Steps: {steps}")
        print(f"Route: {route}")
        
        # Visualize route
        plt = visualize_route(locations, route, f"Evaluation Episode {episode} Route")
        plt.savefig(f"eval_route_episode_{episode}.png")
        plt.close()
    
    # Calculate average results
    avg_reward = np.mean(results["rewards"])
    avg_distance = np.mean(results["distances"])
    avg_steps = np.mean(results["steps"])
    
    print("\nEvaluation Results:")
    print(f"Average reward: {avg_reward:.2f}")
    print(f"Average distance: {avg_distance:.2f}")
    print(f"Average steps: {avg_steps:.2f}")
    
    return results

def compare_with_random(
    agent_path,
    num_locations=5,
    num_episodes=10,
    use_google_maps=False,
    seed=None
):
    """
    Compare trained agent with random policy.
    
    Args:
        agent_path (str): Path to the trained agent
        num_locations (int): Number of locations (including depot)
        num_episodes (int): Number of evaluation episodes
        use_google_maps (bool): Whether to use Google Maps API for distance matrix
        seed (int): Random seed for reproducibility
    """
    print("Comparing trained agent with random policy")
    
    # Set random seed if provided
    if seed is not None:
        np.random.seed(seed)
    
    # Create fixed set of locations for fair comparison
    locations = [(0, 0)]  # Depot at origin
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
    
    # Create traffic conditions
    traffic_conditions = create_traffic_conditions(distance_matrix)
    
    # Initialize environment
    env = DeliveryEnvironment(locations, distance_matrix, traffic_conditions)
    
    # Initialize agent
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dims=[128, 128],
        learning_rate=0.001,
        gamma=0.99,
        epsilon_start=0.0,  # No exploration during evaluation
        epsilon_end=0.0,
        epsilon_decay=1.0,
        buffer_size=10000,
        batch_size=64,
        target_update_freq=10
    )
    
    # Load trained weights
    agent.load(agent_path)
    
    # Results for trained agent
    trained_rewards = []
    trained_distances = []
    trained_routes = []
    
    # Results for random policy
    random_rewards = []
    random_distances = []
    random_routes = []
    
    for episode in range(1, num_episodes + 1):
        print(f"\nComparison Episode {episode}/{num_episodes}")
        
        # Evaluate trained agent
        print("Evaluating trained agent")
        state = env.reset()
        done = False
        total_reward = 0
        route = [0]  # Start at depot
        
        while not done:
            valid_actions = env.get_valid_actions()
            action = agent.select_action(state, valid_actions)
            next_state, reward, done, info = env.step(action)
            state = next_state
            total_reward += reward
            if info["valid_action"]:
                route.append(action)
        
        trained_rewards.append(total_reward)
        trained_distances.append(env.total_distance)
        trained_routes.append(route)
        
        print(f"Trained agent - Total reward: {total_reward:.2f}, Distance: {env.total_distance:.2f}")
        
        # Evaluate random policy
        print("Evaluating random policy")
        state = env.reset()
        done = False
        total_reward = 0
        route = [0]  # Start at depot
        
        while not done:
            valid_actions = env.get_valid_actions()
            valid_indices = np.where(valid_actions)[0]
            if len(valid_indices) > 0:
                action = np.random.choice(valid_indices)
            else:
                action = np.random.randint(action_dim)
            
            next_state, reward, done, info = env.step(action)
            state = next_state
            total_reward += reward
            if info["valid_action"]:
                route.append(action)
        
        random_rewards.append(total_reward)
        random_distances.append(env.total_distance)
        random_routes.append(route)
        
        print(f"Random policy - Total reward: {total_reward:.2f}, Distance: {env.total_distance:.2f}")
    
    # Calculate average results
    avg_trained_reward = np.mean(trained_rewards)
    avg_trained_distance = np.mean(trained_distances)
    avg_random_reward = np.mean(random_rewards)
    avg_random_distance = np.mean(random_distances)
    
    print("\nComparison Results:")
    print(f"Trained agent - Average reward: {avg_trained_reward:.2f}, Average distance: {avg_trained_distance:.2f}")
    print(f"Random policy - Average reward: {avg_random_reward:.2f}, Average distance: {avg_random_distance:.2f}")
    print(f"Improvement: {((avg_random_distance - avg_trained_distance) / avg_random_distance) * 100:.2f}%")
    
    # Plot comparison
    plt.figure(figsize=(10, 6))
    plt.bar(["Trained Agent", "Random Policy"], [avg_trained_distance, avg_random_distance])
    plt.ylabel("Average Distance")
    plt.title("Performance Comparison")
    plt.savefig("comparison_result.png")
    plt.close()
    
    # Visualize routes for last episode
    plt = visualize_route(locations, trained_routes[-1], "Trained Agent Route")
    plt.savefig("trained_agent_route.png")
    plt.close()
    
    plt = visualize_route(locations, random_routes[-1], "Random Policy Route")
    plt.savefig("random_policy_route.png")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained DQN agent for delivery route optimization")
    parser.add_argument("--agent_path", type=str, required=True, help="Path to the trained agent")
    parser.add_argument("--num_locations", type=int, default=5, help="Number of locations including depot")
    parser.add_argument("--num_episodes", type=int, default=10, help="Number of evaluation episodes")
    parser.add_argument("--use_google_maps", action="store_true", help="Use Google Maps API for distance matrix")
    parser.add_argument("--render", action="store_true", help="Render the environment during evaluation")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--compare", action="store_true", help="Compare with random policy")
    
    args = parser.parse_args()
    
    if args.compare:
        compare_with_random(
            agent_path=args.agent_path,
            num_locations=args.num_locations,
            num_episodes=args.num_episodes,
            use_google_maps=args.use_google_maps,
            seed=args.seed
        )
    else:
        evaluate_agent(
            agent_path=args.agent_path,
            num_locations=args.num_locations,
            num_episodes=args.num_episodes,
            use_google_maps=args.use_google_maps,
            render=args.render,
            seed=args.seed
        )