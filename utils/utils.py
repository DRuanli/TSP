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
    
    # Plot
    plt.figure(figsize=(12, 10))
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=200, 
                          node_color=['red' if i == 0 else 'lightblue' for i in range(len(locations))])
    
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
        connectionstyle='arc3,rad=0.1'
    )
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=12)
    
    # Create a colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, label='Traffic Multiplier', shrink=0.8)
    
    # Add title and adjust layout
    plt.title(title)
    plt.axis('off')
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