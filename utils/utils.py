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


# Add these functions to your existing utils.py file

def visualize_traffic(traffic_matrix, locations, title="Traffic Conditions"):
    """
    Visualize traffic conditions as a heatmap overlay on the route network.
    
    Args:
        traffic_matrix (np.ndarray): Matrix of traffic multipliers
        locations (list): List of location coordinates
        title (str): Title for the plot
        
    Returns:
        matplotlib.pyplot: Plot object
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import networkx as nx
    import matplotlib.colors as colors
    
    # Create a graph
    G = nx.DiGraph()
    
    # Add nodes (locations)
    for i, loc in enumerate(locations):
        G.add_node(i, pos=loc)
    
    # Add edges with traffic as edge weights
    n_locations = len(locations)
    for i in range(n_locations):
        for j in range(n_locations):
            if i != j:  # Don't add self-loops
                G.add_edge(i, j, weight=traffic_matrix[i, j])
    
    # Get node positions
    pos = nx.get_node_attributes(G, 'pos')
    
    # Create figure and axis explicitly
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=200, 
                          node_color=['red' if i == 0 else 'lightblue' for i in range(len(locations))],
                          ax=ax)
    
    # Draw edges with traffic-based coloring
    edge_colors = [G[u][v]['weight'] for u, v in G.edges()]
    
    # Create colormap (green to red) for traffic intensity
    cmap = plt.cm.RdYlGn_r  # Red-Yellow-Green color map (reversed)
    
    # Normalize colors
    min_traffic = min(edge_colors)
    max_traffic = max(max(edge_colors), min_traffic + 0.1)  # Ensure range isn't zero
    norm = colors.Normalize(vmin=min_traffic, vmax=max_traffic)
    
    edges = nx.draw_networkx_edges(
        G, pos, 
        edge_color=edge_colors,
        edge_cmap=cmap,
        edge_vmin=min_traffic,
        edge_vmax=max_traffic,
        width=2.0,
        arrowsize=10,
        arrowstyle='-|>',
        connectionstyle='arc3,rad=0.1',
        ax=ax
    )
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=12, ax=ax)
    
    # Create a colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label='Traffic Multiplier', shrink=0.8)
    
    # Add title and adjust layout
    ax.set_title(title)
    ax.axis('off')
    plt.tight_layout()
    
    return plt

def visualize_route_with_traffic(locations, route, traffic_matrix, title="Route with Traffic"):
    """
    Visualize the delivery route with traffic conditions.
    
    Args:
        locations (list): List of location coordinates
        route (list): List of location indices in the order of visitation
        traffic_matrix (np.ndarray): Matrix of traffic multipliers
        title (str): Title for the plot
        
    Returns:
        matplotlib.pyplot: Plot object
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import networkx as nx
    import matplotlib.colors as colors
    
    # Create a graph
    G = nx.DiGraph()
    
    # Add nodes (locations)
    for i, loc in enumerate(locations):
        G.add_node(i, pos=loc)
    
    # Add edges for the entire network with traffic as edge weights
    n_locations = len(locations)
    for i in range(n_locations):
        for j in range(n_locations):
            if i != j:  # Don't add self-loops
                G.add_edge(i, j, weight=traffic_matrix[i, j], in_route=False)
    
    # Mark edges in the route
    for i in range(len(route) - 1):
        from_node = route[i]
        to_node = route[i + 1]
        G[from_node][to_node]['in_route'] = True
    
    # Get node positions
    pos = nx.get_node_attributes(G, 'pos')
    
    # Plot
    plt.figure(figsize=(12, 10))
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=200, 
                          node_color=['red' if i == 0 else 'lightblue' for i in range(len(locations))])
    
    # Draw non-route edges (background network)
    non_route_edges = [(u, v) for u, v in G.edges() if not G[u][v]['in_route']]
    nx.draw_networkx_edges(
        G, pos, 
        edgelist=non_route_edges,
        edge_color='gray',
        alpha=0.2,
        width=1.0,
        arrows=False
    )
    
    # Draw route edges with traffic-based coloring
    route_edges = [(u, v) for u, v in G.edges() if G[u][v]['in_route']]
    edge_colors = [G[u][v]['weight'] for u, v in route_edges]
    
    # Create colormap (green to red) for traffic intensity
    cmap = plt.cm.RdYlGn_r  # Red-Yellow-Green color map (reversed)
    
    # Normalize colors
    min_traffic = min(edge_colors) if edge_colors else 1.0
    max_traffic = max(max(edge_colors), min_traffic + 0.1) if edge_colors else 2.0  # Ensure range isn't zero
    norm = colors.Normalize(vmin=min_traffic, vmax=max_traffic)
    
    edges = nx.draw_networkx_edges(
        G, pos, 
        edgelist=route_edges,
        edge_color=edge_colors,
        edge_cmap=cmap,
        edge_vmin=min_traffic,
        edge_vmax=max_traffic,
        width=2.5,
        arrowsize=15,
        arrowstyle='-|>',
        connectionstyle='arc3,rad=0.1'
    )
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=12)
    
    # Add route order labels
    edge_labels = {(route[i], route[i+1]): f"{i+1}" for i in range(len(route)-1)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='blue')
    
    # Create a colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, label='Traffic Multiplier', shrink=0.8)
    
    # Add title and adjust layout
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    
    return plt

def visualize_traffic_change(before_traffic, after_traffic, locations, title="Traffic Change"):
    """
    Visualize the change in traffic conditions.
    
    Args:
        before_traffic (np.ndarray): Traffic matrix before update
        after_traffic (np.ndarray): Traffic matrix after update
        locations (list): List of location coordinates
        title (str): Title for the plot
        
    Returns:
        matplotlib.pyplot: Plot object
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import networkx as nx
    import matplotlib.colors as colors
    
    # Create a graph
    G = nx.DiGraph()
    
    # Add nodes (locations)
    for i, loc in enumerate(locations):
        G.add_node(i, pos=loc)
    
    # Calculate traffic changes
    traffic_change = after_traffic - before_traffic
    
    # Add edges with traffic change as edge weights
    n_locations = len(locations)
    for i in range(n_locations):
        for j in range(n_locations):
            if i != j:  # Don't add self-loops
                change = traffic_change[i, j]
                # Only add edges where traffic changed
                if abs(change) > 0.01:
                    G.add_edge(i, j, weight=change)
    
    # Get node positions
    pos = nx.get_node_attributes(G, 'pos')
    
    # Plot
    plt.figure(figsize=(12, 10))
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=200, 
                          node_color=['red' if i == 0 else 'lightblue' for i in range(len(locations))])
    
    # Draw edges with traffic change coloring
    edge_colors = [G[u][v]['weight'] for u, v in G.edges()]
    
    if not edge_colors:
        # No traffic changes to visualize
        plt.title(f"{title}\n(No significant changes)")
        plt.axis('off')
        plt.tight_layout()
        return plt
    
    # Create colormap (blue-white-red) for traffic changes
    cmap = plt.cm.RdBu
    
    # Normalize colors - symmetric around zero
    max_abs_change = max(abs(min(edge_colors)), abs(max(edge_colors)))
    norm = colors.Normalize(vmin=-max_abs_change, vmax=max_abs_change)
    
    edges = nx.draw_networkx_edges(
        G, pos, 
        edge_color=edge_colors,
        edge_cmap=cmap,
        edge_vmin=-max_abs_change,
        edge_vmax=max_abs_change,
        width=2.0,
        arrowsize=10,
        arrowstyle='-|>',
        connectionstyle='arc3,rad=0.1'
    )
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=12)
    
    # Create a colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, label='Traffic Change (+ = worse, - = better)', shrink=0.8)
    
    # Add title and adjust layout
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    
    return plt