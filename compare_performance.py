import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import argparse

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from evaluate import evaluate_agent
from evaluate_dynamic import evaluate_dynamic_agent

def compare_performance(
    static_agent_path,
    dynamic_agent_path,
    num_locations=5,
    num_episodes=10,
    use_google_maps=False,
    traffic_update_freq=5,
    traffic_significance_threshold=0.2,
    include_traffic_in_state=True,
    traffic_intensity=0.3,
    traffic_incident_probability=0.05,
    seed=None,
    output_dir="comparison_results"
):
    """
    Compare performance of static vs dynamic agents under different traffic scenarios.
    
    Args:
        static_agent_path (str): Path to the static agent
        dynamic_agent_path (str): Path to the dynamic agent
        num_locations (int): Number of locations (including depot)
        num_episodes (int): Number of evaluation episodes
        use_google_maps (bool): Whether to use Google Maps API for distance matrix
        traffic_update_freq (int): Frequency of traffic updates in steps
        traffic_significance_threshold (float): Minimum change to be considered significant
        include_traffic_in_state (bool): Whether to include traffic info in state
        traffic_intensity (float): Intensity of traffic variations
        traffic_incident_probability (float): Probability of traffic incidents
        seed (int): Random seed for reproducibility
        output_dir (str): Directory to save comparison results
    """
    print(f"Comparing static vs dynamic agents under different traffic scenarios")
    
    # Set random seed if provided
    if seed is not None:
        np.random.seed(seed)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Traffic scenarios to test
    traffic_scenarios = [
        {"name": "No Traffic", "intensity": 0.0, "incident_prob": 0.0},
        {"name": "Light Traffic", "intensity": 0.1, "incident_prob": 0.02},
        {"name": "Moderate Traffic", "intensity": 0.3, "incident_prob": 0.05},
        {"name": "Heavy Traffic", "intensity": 0.5, "incident_prob": 0.1},
        {"name": "Extreme Traffic", "intensity": 0.7, "incident_prob": 0.15}
    ]
    
    # Results storage
    comparison_results = {
        "scenario_names": [],
        "static_rewards": [],
        "static_distances": [],
        "dynamic_rewards": [],
        "dynamic_distances": [],
        "reward_improvements": [],
        "distance_improvements": []
    }
    
    # Compare performance across scenarios
    for scenario in traffic_scenarios:
        print(f"\n*** Testing Scenario: {scenario['name']} ***")
        print(f"Traffic intensity: {scenario['intensity']}")
        print(f"Incident probability: {scenario['incident_prob']}")
        
        # Use the same seed for both evaluations to ensure fair comparison
        scenario_seed = np.random.randint(10000) if seed is None else seed
        
        # Evaluate static agent
        print("\nEvaluating static agent...")
        static_results = evaluate_agent(
            agent_path=static_agent_path,
            num_locations=num_locations,
            num_episodes=num_episodes,
            use_google_maps=use_google_maps,
            render=False,
            seed=scenario_seed
        )
        
        # Evaluate dynamic agent
        print("\nEvaluating dynamic agent...")
        dynamic_results = evaluate_dynamic_agent(
            agent_path=dynamic_agent_path,
            num_locations=num_locations,
            num_episodes=num_episodes,
            use_google_maps=use_google_maps,
            traffic_update_freq=traffic_update_freq,
            traffic_significance_threshold=traffic_significance_threshold,
            include_traffic_in_state=include_traffic_in_state,
            traffic_intensity=scenario["intensity"],
            traffic_incident_probability=scenario["incident_prob"],
            render=False,
            seed=scenario_seed
        )
        
        # Calculate averages
        avg_static_reward = np.mean(static_results["rewards"])
        avg_static_distance = np.mean(static_results["distances"])
        avg_dynamic_reward = np.mean(dynamic_results["rewards"])
        avg_dynamic_distance = np.mean(dynamic_results["distances"])
        
        # Calculate improvement percentages
        reward_improvement = ((avg_dynamic_reward - avg_static_reward) / abs(avg_static_reward)) * 100
        distance_improvement = ((avg_static_distance - avg_dynamic_distance) / avg_static_distance) * 100
        
        # Store results
        comparison_results["scenario_names"].append(scenario["name"])
        comparison_results["static_rewards"].append(avg_static_reward)
        comparison_results["static_distances"].append(avg_static_distance)
        comparison_results["dynamic_rewards"].append(avg_dynamic_reward)
        comparison_results["dynamic_distances"].append(avg_dynamic_distance)
        comparison_results["reward_improvements"].append(reward_improvement)
        comparison_results["distance_improvements"].append(distance_improvement)
        
        # Print summary
        print(f"\nScenario: {scenario['name']} - Summary:")
        print(f"Static agent - Avg reward: {avg_static_reward:.2f}, Avg distance: {avg_static_distance:.2f}")
        print(f"Dynamic agent - Avg reward: {avg_dynamic_reward:.2f}, Avg distance: {avg_dynamic_distance:.2f}")
        print(f"Reward improvement: {reward_improvement:.2f}%")
        print(f"Distance improvement: {distance_improvement:.2f}%")
    
    # Plot reward comparison
    plt.figure(figsize=(14, 7))
    x = np.arange(len(comparison_results["scenario_names"]))
    width = 0.35
    
    plt.bar(x - width/2, comparison_results["static_rewards"], width, label='Static Agent')
    plt.bar(x + width/2, comparison_results["dynamic_rewards"], width, label='Dynamic Agent')
    
    plt.xlabel('Traffic Scenario')
    plt.ylabel('Average Reward')
    plt.title('Reward Comparison Across Traffic Scenarios')
    plt.xticks(x, comparison_results["scenario_names"])
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add improvement percentages as text
    for i, improvement in enumerate(comparison_results["reward_improvements"]):
        plt.text(i, min(comparison_results["static_rewards"][i], comparison_results["dynamic_rewards"][i]), 
                f'{improvement:.1f}%', ha='center', va='bottom', color='blue', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "reward_comparison.png"))
    plt.close()
    
    # Plot distance comparison
    plt.figure(figsize=(14, 7))
    
    plt.bar(x - width/2, comparison_results["static_distances"], width, label='Static Agent')
    plt.bar(x + width/2, comparison_results["dynamic_distances"], width, label='Dynamic Agent')
    
    plt.xlabel('Traffic Scenario')
    plt.ylabel('Average Distance')
    plt.title('Distance Comparison Across Traffic Scenarios')
    plt.xticks(x, comparison_results["scenario_names"])
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add improvement percentages as text
    for i, improvement in enumerate(comparison_results["distance_improvements"]):
        plt.text(i, max(comparison_results["static_distances"][i], comparison_results["dynamic_distances"][i]), 
                f'{improvement:.1f}%', ha='center', va='top', color='green', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "distance_comparison.png"))
    plt.close()
    
    # Plot improvement percentages
    plt.figure(figsize=(14, 7))
    
    plt.bar(x - width/2, comparison_results["reward_improvements"], width, label='Reward Improvement')
    plt.bar(x + width/2, comparison_results["distance_improvements"], width, label='Distance Improvement')
    
    plt.xlabel('Traffic Scenario')
    plt.ylabel('Improvement (%)')
    plt.title('Performance Improvement of Dynamic vs Static Agent')
    plt.xticks(x, comparison_results["scenario_names"])
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add zero line
    plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "improvement_comparison.png"))
    plt.close()
    
    # Save results to CSV
    import pandas as pd
    results_df = pd.DataFrame({
        'Scenario': comparison_results["scenario_names"],
        'Static_Reward': comparison_results["static_rewards"],
        'Dynamic_Reward': comparison_results["dynamic_rewards"],
        'Reward_Improvement(%)': comparison_results["reward_improvements"],
        'Static_Distance': comparison_results["static_distances"],
        'Dynamic_Distance': comparison_results["dynamic_distances"],
        'Distance_Improvement(%)': comparison_results["distance_improvements"]
    })
    
    results_df.to_csv(os.path.join(output_dir, "comparison_results.csv"), index=False)
    
    print(f"\nComparison complete. Results saved to {output_dir}")
    return comparison_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare static and dynamic DQN agents")
    parser.add_argument("--static_agent_path", type=str, required=True, help="Path to the static agent")
    parser.add_argument("--dynamic_agent_path", type=str, required=True, help="Path to the dynamic agent")
    parser.add_argument("--num_locations", type=int, default=5, help="Number of locations including depot")
    parser.add_argument("--num_episodes", type=int, default=5, help="Number of evaluation episodes per scenario")
    parser.add_argument("--use_google_maps", action="store_true", help="Use Google Maps API for distance matrix")
    parser.add_argument("--traffic_update_freq", type=int, default=5, help="Frequency of traffic updates in steps")
    parser.add_argument("--traffic_significance_threshold", type=float, default=0.2, help="Minimum change to be considered significant")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--output_dir", type=str, default="comparison_results", help="Directory to save comparison results")
    
    args = parser.parse_args()
    
    compare_performance(
        static_agent_path=args.static_agent_path,
        dynamic_agent_path=args.dynamic_agent_path,
        num_locations=args.num_locations,
        num_episodes=args.num_episodes,
        use_google_maps=args.use_google_maps,
        traffic_update_freq=args.traffic_update_freq,
        traffic_significance_threshold=args.traffic_significance_threshold,
        seed=args.seed,
        output_dir=args.output_dir
    )