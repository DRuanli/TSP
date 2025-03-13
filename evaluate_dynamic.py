import numpy as np
import matplotlib.pyplot as plt
import torch
import os
import sys
import argparse

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from environment.dynamic_delivery_env import DynamicDeliveryEnvironment
from agents.dynamic_dqn_agent import DynamicDQNAgent
from utils.utils import create_distance_matrix, create_traffic_conditions, visualize_route, visualize_traffic
from utils.google_maps_api import create_distance_matrix_with_api

def evaluate_dynamic_agent(
    agent_path,
    num_locations=5,
    num_episodes=10,
    use_google_maps=False,
    traffic_update_freq=5,
    traffic_significance_threshold=0.2,
    include_traffic_in_state=True,
    traffic_intensity=0.3,
    traffic_incident_probability=0.05,
    render=True,
    seed=None
):
    """
    Evaluate a trained dynamic DQN agent on new sets of cities with dynamic traffic.
    
    Args:
        agent_path (str): Path to the trained agent
        num_locations (int): Number of locations (including depot)
        num_episodes (int): Number of evaluation episodes
        use_google_maps (bool): Whether to use Google Maps API for distance matrix
        traffic_update_freq (int): Frequency of traffic updates in steps
        traffic_significance_threshold (float): Minimum change to be considered significant
        include_traffic_in_state (bool): Whether to include traffic info in state
        traffic_intensity (float): Intensity of traffic variations
        traffic_incident_probability (float): Probability of traffic incidents
        render (bool): Whether to render the environment during evaluation
        seed (int): Random seed for reproducibility
    
    Returns:
        dict: Evaluation results
    """
    print(f"Evaluating dynamic agent from {agent_path}")
    
    # Set random seed if provided
    if seed is not None:
        np.random.seed(seed)
    
    # Initialize results
    results = {
        "rewards": [],
        "distances": [],
        "steps": [],
        "routes": [],
        "traffic_updates": [],
        "significant_updates": [],
        "replanning_events": []
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
        traffic_conditions = create_traffic_conditions(distance_matrix, congestion_factor=traffic_intensity)
        
        # Initialize environment
        env = DynamicDeliveryEnvironment(
            locations, 
            distance_matrix, 
            traffic_conditions,
            traffic_update_freq=traffic_update_freq,
            traffic_significance_threshold=traffic_significance_threshold,
            include_traffic_in_state=include_traffic_in_state
        )
        
        # Initialize agent
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n
        
        agent = DynamicDQNAgent(
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
            target_update_freq=10,
            prioritized_replay=True
        )
        
        # Load trained weights
        agent.load(agent_path)
        
        # Reset environment
        state = env.reset()
        done = False
        total_reward = 0
        steps = 0
        route = [0]  # Start at depot
        traffic_updates = 0
        significant_updates = 0
        replanning_events = 0
        
        # For visualization of traffic changes
        traffic_snapshots = [env.get_traffic_heatmap().copy()]
        
        while not done:
            # Get valid actions
            valid_actions = env.get_valid_actions()
            
            # Check if traffic has significantly changed
            traffic_changed = hasattr(env, 'significant_update') and env.significant_update
            
            # Select action (no exploration)
            action = agent.select_action(state, valid_actions, traffic_changed)
            
            # Take action
            next_state, reward, done, info = env.step(action)
            
            # Track traffic updates
            if info.get("traffic_updated", False):
                traffic_updates += 1
                # Save traffic snapshot for visualization
                traffic_snapshots.append(env.get_traffic_heatmap().copy())
                
            if info.get("significant_traffic_update", False):
                significant_updates += 1
                # Implement replanning
                did_replan = agent.replan_if_needed(env, True)
                if did_replan:
                    replanning_events += 1
            
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
        results["traffic_updates"].append(traffic_updates)
        results["significant_updates"].append(significant_updates)
        results["replanning_events"].append(replanning_events)
        
        print(f"Total reward: {total_reward:.2f}")
        print(f"Total distance: {env.total_distance:.2f}")
        print(f"Steps: {steps}")
        print(f"Traffic updates: {traffic_updates}")
        print(f"Significant traffic changes: {significant_updates}")
        print(f"Replanning events: {replanning_events}")
        print(f"Route: {route}")
        
        # Visualize route
        plt = visualize_route(locations, route, f"Dynamic Evaluation Episode {episode} Route")
        plt.savefig(f"eval_dynamic_route_episode_{episode}.png")
        plt.close()
        
        # Visualize final traffic
        plt = visualize_traffic(env.get_traffic_heatmap(), locations, f"Final Traffic State Episode {episode}")
        plt.savefig(f"eval_traffic_episode_{episode}.png")
        plt.close()
        
        # Create traffic evolution animation (simplified as sequence of images)
        for i, traffic in enumerate(traffic_snapshots):
            plt = visualize_traffic(traffic, locations, f"Traffic State {i}")
            plt.savefig(f"traffic_evolution_ep{episode}_step{i}.png")
            plt.close()
    
    # Calculate average results
    avg_reward = np.mean(results["rewards"])
    avg_distance = np.mean(results["distances"])
    avg_steps = np.mean(results["steps"])
    avg_traffic_updates = np.mean(results["traffic_updates"])
    avg_significant_updates = np.mean(results["significant_updates"])
    avg_replanning_events = np.mean(results["replanning_events"])
    
    print("\nEvaluation Results:")
    print(f"Average reward: {avg_reward:.2f}")
    print(f"Average distance: {avg_distance:.2f}")
    print(f"Average steps: {avg_steps:.2f}")
    print(f"Average traffic updates: {avg_traffic_updates:.2f}")
    print(f"Average significant updates: {avg_significant_updates:.2f}")
    print(f"Average replanning events: {avg_replanning_events:.2f}")
    
    return results

def compare_static_vs_dynamic(
    static_agent_path,
    dynamic_agent_path,
    num_locations=5,
    num_episodes=10,
    use_google_maps=False,
    traffic_update_freq=5,
    traffic_significance_threshold=0.2,
    include_traffic_in_state=True,
    seed=None
):
    """
    Compare static DQN agent with dynamic DQN agent under changing traffic conditions.
    
    Args:
        static_agent_path (str): Path to the static agent
        dynamic_agent_path (str): Path to the dynamic agent
        num_locations (int): Number of locations (including depot)
        num_episodes (int): Number of evaluation episodes
        use_google_maps (bool): Whether to use Google Maps API for distance matrix
        traffic_update_freq (int): Frequency of traffic updates in steps
        traffic_significance_threshold (float): Minimum change to be considered significant
        include_traffic_in_state (bool): Whether to include traffic info in state
        seed (int): Random seed for reproducibility
    """
    print("Comparing static vs dynamic agents with changing traffic")
    
    # Set random seed if provided
    if seed is not None:
        np.random.seed(seed)
    
    # Results for static agent
    static_rewards = []
    static_distances = []
    static_routes = []
    
    # Results for dynamic agent
    dynamic_rewards = []
    dynamic_distances = []
    dynamic_routes = []
    
    for episode in range(1, num_episodes + 1):
        print(f"\nComparison Episode {episode}/{num_episodes}")
        
        # Create same set of locations for both agents
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
        
        # Create a traffic simulator seed to ensure both agents encounter the same traffic
        traffic_seed = np.random.randint(10000)
        
        # Evaluate static agent
        print("Evaluating static agent")
        np.random.seed(traffic_seed)  # Use same traffic seed
        
        # Initialize environment
        env_static = DynamicDeliveryEnvironment(
            locations, 
            distance_matrix, 
            traffic_conditions,
            traffic_update_freq=traffic_update_freq,
            traffic_significance_threshold=traffic_significance_threshold,
            include_traffic_in_state=False  # Static agent doesn't use traffic in state
        )
        
        # Initialize static agent
        state_dim_static = env_static.observation_space.shape[0]
        action_dim = env_static.action_space.n
        
        from agents.dqn_agent import DQNAgent  # Import the static agent
        
        static_agent = DQNAgent(
            state_dim=state_dim_static,
            action_dim=action_dim,
            hidden_dims=[128, 128],
            epsilon_start=0.0,  # No exploration during evaluation
            epsilon_end=0.0,
            epsilon_decay=1.0
        )
        
        # Load trained weights
        static_agent.load(static_agent_path)
        
        # Evaluate
        state = env_static.reset()
        done = False
        total_reward = 0
        route = [0]  # Start at depot
        
        while not done:
            valid_actions = env_static.get_valid_actions()
            action = static_agent.select_action(state, valid_actions)
            next_state, reward, done, info = env_static.step(action)
            state = next_state
            total_reward += reward
            if info["valid_action"]:
                route.append(action)
        
        static_rewards.append(total_reward)
        static_distances.append(env_static.total_distance)
        static_routes.append(route)
        
        print(f"Static agent - Total reward: {total_reward:.2f}, Distance: {env_static.total_distance:.2f}")
        
        # Evaluate dynamic agent
        print("Evaluating dynamic agent")
        np.random.seed(traffic_seed)  # Use same traffic seed
        
        # Initialize environment
        env_dynamic = DynamicDeliveryEnvironment(
            locations, 
            distance_matrix, 
            traffic_conditions,
            traffic_update_freq=traffic_update_freq,
            traffic_significance_threshold=traffic_significance_threshold,
            include_traffic_in_state=include_traffic_in_state
        )
        
        # Initialize dynamic agent
        state_dim_dynamic = env_dynamic.observation_space.shape[0]
        
        dynamic_agent = DynamicDQNAgent(
            state_dim=state_dim_dynamic,
            action_dim=action_dim,
            hidden_dims=[128, 128],
            epsilon_start=0.0,  # No exploration during evaluation
            epsilon_end=0.0,
            epsilon_decay=1.0,
            prioritized_replay=True
        )
        
        # Load trained weights
        dynamic_agent.load(dynamic_agent_path)
        
        # Evaluate
        state = env_dynamic.reset()
        done = False
        total_reward = 0
        route = [0]  # Start at depot
        
        while not done:
            valid_actions = env_dynamic.get_valid_actions()
            traffic_changed = hasattr(env_dynamic, 'significant_update') and env_dynamic.significant_update
            action = dynamic_agent.select_action(state, valid_actions, traffic_changed)
            next_state, reward, done, info = env_dynamic.step(action)
            
            # Handle significant traffic updates
            if info.get("significant_traffic_update", False):
                dynamic_agent.replan_if_needed(env_dynamic, True)
            
            state = next_state
            total_reward += reward
            if info["valid_action"]:
                route.append(action)
        
        dynamic_rewards.append(total_reward)
        dynamic_distances.append(env_dynamic.total_distance)
        dynamic_routes.append(route)
        
        print(f"Dynamic agent - Total reward: {total_reward:.2f}, Distance: {env_dynamic.total_distance:.2f}")
        
        # Visualize routes for this episode
        plt = visualize_route(locations, static_routes[-1], f"Static Agent Route - Episode {episode}")
        plt.savefig(f"static_route_episode_{episode}.png")
        plt.close()
        
        plt = visualize_route(locations, dynamic_routes[-1], f"Dynamic Agent Route - Episode {episode}")
        plt.savefig(f"dynamic_route_episode_{episode}.png")
        plt.close()
    
    # Calculate average results
    avg_static_reward = np.mean(static_rewards)
    avg_static_distance = np.mean(static_distances)
    avg_dynamic_reward = np.mean(dynamic_rewards)
    avg_dynamic_distance = np.mean(dynamic_distances)
    
    # Calculate improvement
    reward_improvement = ((avg_dynamic_reward - avg_static_reward) / abs(avg_static_reward)) * 100
    distance_improvement = ((avg_static_distance - avg_dynamic_distance) / avg_static_distance) * 100
    
    print("\nComparison Results:")
    print(f"Static agent - Average reward: {avg_static_reward:.2f}, Average distance: {avg_static_distance:.2f}")
    print(f"Dynamic agent - Average reward: {avg_dynamic_reward:.2f}, Average distance: {avg_dynamic_distance:.2f}")
    print(f"Reward improvement: {reward_improvement:.2f}%")
    print(f"Distance improvement: {distance_improvement:.2f}%")
    
    # Plot comparison
    labels = ["Static Agent", "Dynamic Agent"]
    rewards = [avg_static_reward, avg_dynamic_reward]
    distances = [avg_static_distance, avg_dynamic_distance]
    
    # Plot rewards
    plt.figure(figsize=(10, 6))
    plt.bar(labels, rewards)
    plt.ylabel("Average Reward")
    plt.title("Reward Comparison")
    plt.savefig("comparison_rewards.png")
    plt.close()
    
    # Plot distances
    plt.figure(figsize=(10, 6))
    plt.bar(labels, distances)
    plt.ylabel("Average Distance")
    plt.title("Distance Comparison")
    plt.savefig("comparison_distances.png")
    plt.close()
    
    return {
        "static_rewards": static_rewards,
        "static_distances": static_distances,
        "dynamic_rewards": dynamic_rewards,
        "dynamic_distances": dynamic_distances,
        "reward_improvement": reward_improvement,
        "distance_improvement": distance_improvement
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate dynamic DQN agent for delivery route optimization")
    parser.add_argument("--agent_path", type=str, required=True, help="Path to the trained agent")
    parser.add_argument("--num_locations", type=int, default=5, help="Number of locations including depot")
    parser.add_argument("--num_episodes", type=int, default=10, help="Number of evaluation episodes")
    parser.add_argument("--use_google_maps", action="store_true", help="Use Google Maps API for distance matrix")
    parser.add_argument("--traffic_update_freq", type=int, default=5, help="Frequency of traffic updates in steps")
    parser.add_argument("--traffic_significance_threshold", type=float, default=0.2, help="Minimum change to be considered significant")
    parser.add_argument("--include_traffic_in_state", action="store_true", help="Include traffic info in state")
    parser.add_argument("--traffic_intensity", type=float, default=0.3, help="Intensity of traffic variations")
    parser.add_argument("--render", action="store_true", help="Render the environment during evaluation")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--compare", action="store_true", help="Compare with static agent")
    parser.add_argument("--static_agent_path", type=str, help="Path to the static agent (needed for comparison)")
    
    args = parser.parse_args()
    
    if args.compare:
        if args.static_agent_path is None:
            print("Error: --static_agent_path is required for comparison")
            sys.exit(1)
            
        compare_static_vs_dynamic(
            static_agent_path=args.static_agent_path,
            dynamic_agent_path=args.agent_path,
            num_locations=args.num_locations,
            num_episodes=args.num_episodes,
            use_google_maps=args.use_google_maps,
            traffic_update_freq=args.traffic_update_freq,
            traffic_significance_threshold=args.traffic_significance_threshold,
            include_traffic_in_state=args.include_traffic_in_state,
            seed=args.seed
        )
    else:
        evaluate_dynamic_agent(
            agent_path=args.agent_path,
            num_locations=args.num_locations,
            num_episodes=args.num_episodes,
            use_google_maps=args.use_google_maps,
            traffic_update_freq=args.traffic_update_freq,
            traffic_significance_threshold=args.traffic_significance_threshold,
            include_traffic_in_state=args.include_traffic_in_state,
            traffic_intensity=args.traffic_intensity,
            render=args.render,
            seed=args.seed
        )