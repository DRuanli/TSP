import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


def create_distance_matrix(locations, is_euclidean=True):
    """
    Create a distance matrix from locations.
    
    Args:
        locations (list): List of location coordinates (e.g., [(lat1, lon1), (lat2, lon2), ...])
        is_euclidean (bool): Whether to use Euclidean distance or simulated travel time
        
    Returns:
        np.ndarray: Distance/travel time matrix
    """
    n_locations = len(locations)
    distance_matrix = np.zeros((n_locations, n_locations))
    
    for i in range(n_locations):
        for j in range(n_locations):
            if i == j:
                distance_matrix[i, j] = 0
            else:
                if is_euclidean:
                    # Calculate Euclidean distance
                    loc1 = np.array(locations[i])
                    loc2 = np.array(locations[j])
                    distance_matrix[i, j] = np.linalg.norm(loc1 - loc2)
                else:
                    # Simulate travel time (could be replaced with actual API calls)
                    # Here we use a simple model: time = distance * (1 + random noise)
                    loc1 = np.array(locations[i])
                    loc2 = np.array(locations[j])
                    distance = np.linalg.norm(loc1 - loc2)
                    noise = np.random.uniform(0.8, 1.2)  # Random noise to simulate variability
                    distance_matrix[i, j] = distance * noise
    
    return distance_matrix


def create_traffic_conditions(distance_matrix, congestion_factor=0.3):
    """
    Create traffic conditions matrix.
    
    Args:
        distance_matrix (np.ndarray): Distance/travel time matrix
        congestion_factor (float): Factor to determine traffic variation (0 = no traffic)
        
    Returns:
        np.ndarray: Traffic multiplier matrix (1.0 = normal, >1.0 = congestion)
    """
    n_locations = distance_matrix.shape[0]
    traffic_matrix = np.ones_like(distance_matrix)
    
    # Randomly select roads with congestion
    for i in range(n_locations):
        for j in range(n_locations):
            if i != j:
                # Randomly assign traffic conditions
                if np.random.random() < congestion_factor:
                    # Congestion: multiply travel time by a factor > 1
                    traffic_matrix[i, j] = np.random.uniform(1.1, 2.0)
    
    return traffic_matrix


def visualize_route(locations, route, title="Delivery Route"):
    """
    Visualize the delivery route.
    
    Args:
        locations (list): List of location coordinates
        route (list): List of location indices in the order of visitation
        title (str): Title for the plot
    """
    # Create a graph
    G = nx.DiGraph()
    
    # Add nodes (locations)
    for i, loc in enumerate(locations):
        G.add_node(i, pos=loc)
    
    # Add edges (route)
    for i in range(len(route) - 1):
        G.add_edge(route[i], route[i + 1])
    
    # Get node positions
    pos = nx.get_node_attributes(G, 'pos')
    
    # Plot
    plt.figure(figsize=(10, 8))
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=200, 
                          node_color=['red' if i == 0 else 'lightblue' for i in range(len(locations))])
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, edgelist=G.edges(), arrows=True)
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=12)
    
    # Add title and adjust layout
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    
    return plt


def plot_learning_curve(episode_rewards, window_size=10):
    """
    Plot the learning curve (rewards over episodes).
    
    Args:
        episode_rewards (list): List of rewards for each episode
        window_size (int): Window size for smoothing
    """
    # Calculate moving average for smoothing
    if len(episode_rewards) >= window_size:
        smoothed_rewards = [np.mean(episode_rewards[max(0, i - window_size):i]) 
                          for i in range(1, len(episode_rewards) + 1)]
    else:
        smoothed_rewards = episode_rewards
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(episode_rewards, alpha=0.3, label='Raw Rewards')
    plt.plot(smoothed_rewards, label=f'Smoothed Rewards (window={window_size})')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Learning Curve')
    plt.legend()
    plt.grid(True)
    
    return plt


def mock_google_maps_api(origin_location, destination_location, traffic_factor=1.0):
    """
    Mock function to simulate Google Maps API for travel time estimation.
    
    Args:
        origin_location (tuple): Coordinates of origin (lat, lon)
        destination_location (tuple): Coordinates of destination (lat, lon)
        traffic_factor (float): Traffic multiplier (1.0 = normal, >1.0 = congestion)
        
    Returns:
        float: Estimated travel time in minutes
    """
    # Calculate Euclidean distance
    origin = np.array(origin_location)
    destination = np.array(destination_location)
    distance = np.linalg.norm(origin - destination)
    
    # Convert to simulated travel time (minutes)
    # Assume average speed of 60 distance units per hour
    travel_time = (distance / 60) * 60  # Convert to minutes
    
    # Apply traffic factor
    travel_time *= traffic_factor
    
    # Add some random noise
    noise = np.random.uniform(0.9, 1.1)
    travel_time *= noise
    
    return travel_time