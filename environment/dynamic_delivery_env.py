import numpy as np
import gym
from gym import spaces
import networkx as nx
from environment.delivery_env import DeliveryEnvironment
from utils.traffic_simulator import TrafficSimulator


class DynamicDeliveryEnvironment(DeliveryEnvironment):
    """
    Environment for the Delivery Route Optimization Problem with dynamic traffic updates.
    Extends the base DeliveryEnvironment to include real-time traffic changes.
    """
    
    def __init__(
        self,
        locations,
        distance_matrix,
        traffic_conditions=None,
        traffic_update_freq=5,
        traffic_significance_threshold=0.2,
        include_traffic_in_state=True
    ):
        """
        Initialize the Dynamic Delivery Environment.
        
        Args:
            locations (list): List of location coordinates (e.g., [(lat1, lon1), (lat2, lon2), ...])
            distance_matrix (np.ndarray): Matrix of distances/travel times between locations
            traffic_conditions (np.ndarray, optional): Initial traffic multipliers
            traffic_update_freq (int): Frequency of traffic updates in steps
            traffic_significance_threshold (float): Minimum change to be considered significant
            include_traffic_in_state (bool): Whether to include traffic info in state
        """
        # Initialize the base environment
        super(DynamicDeliveryEnvironment, self).__init__(
            locations, distance_matrix, traffic_conditions
        )
        
        # Traffic update parameters
        self.traffic_update_freq = traffic_update_freq
        self.traffic_significance_threshold = traffic_significance_threshold
        self.include_traffic_in_state = include_traffic_in_state
        
        # Initialize traffic simulator
        self.traffic_simulator = TrafficSimulator(
            distance_matrix,
            base_traffic_conditions=traffic_conditions
        )
        
        # Traffic update tracking
        self.steps_since_update = 0
        self.significant_update = False
        
        # Extend state space if including traffic
        if include_traffic_in_state:
            # Add traffic information to state (average traffic for each location)
            traffic_state_dim = self.n_locations
            self.observation_space = spaces.Box(
                low=0, 
                high=float('inf'),
                shape=(self.n_locations * 2 + traffic_state_dim,),
                dtype=np.float32
            )
    
    def reset(self):
        """
        Reset the environment and traffic conditions.
        
        Returns:
            np.ndarray: Initial state representation
        """
        # Reset base environment
        state = super(DynamicDeliveryEnvironment, self).reset()
        
        # Reset traffic simulator
        self.traffic_conditions = self.traffic_simulator.reset()
        
        # Reset traffic update tracking
        self.steps_since_update = 0
        self.significant_update = False
        
        return self._get_state()
    
    def step(self, action):
        """
        Take an action and update environment with potential traffic changes.
        
        Args:
            action (int): The location to visit next (0 to n_locations-1)
            
        Returns:
            tuple: (next_state, reward, done, info)
        """
        # Take action in base environment
        next_state, reward, done, info = super(DynamicDeliveryEnvironment, self).step(action)
        
        # Update traffic conditions if needed
        self.steps_since_update += 1
        if self.steps_since_update >= self.traffic_update_freq:
            self._update_traffic()
            self.steps_since_update = 0
        
        # Update state with current traffic conditions
        next_state = self._get_state()
        
        # Add traffic update info to info dict
        info["traffic_updated"] = self.steps_since_update == 0
        info["significant_traffic_update"] = self.significant_update
        
        return next_state, reward, done, info
    
    def _update_traffic(self):
        """
        Update traffic conditions using the traffic simulator.
        """
        # Get previous traffic for comparison
        previous_traffic = self.traffic_conditions.copy()
        
        # Update traffic conditions
        new_traffic, change_magnitude = self.traffic_simulator.update()
        self.traffic_conditions = new_traffic
        
        # Check if update is significant
        self.significant_update = change_magnitude >= self.traffic_significance_threshold
        
        return change_magnitude
    
    def _get_state(self):
        """
        Get state representation including traffic conditions if enabled.
        
        Returns:
            np.ndarray: State representation
        """
        # Get base state (current location one-hot + visited status)
        base_state = super(DynamicDeliveryEnvironment, self)._get_state()
        
        if not self.include_traffic_in_state:
            return base_state
        
        # Add traffic information
        traffic_state = self._get_traffic_state()
        
        # Combine into expanded state vector
        state = np.concatenate([base_state, traffic_state])
        
        return state
    
    def _get_traffic_state(self):
        """
        Create a representation of current traffic conditions.
        
        Returns:
            np.ndarray: Traffic state representation
        """
        # Simple approach: average traffic condition for each location
        traffic_state = np.zeros(self.n_locations)
        
        for i in range(self.n_locations):
            # Average traffic from this location to all others
            outgoing_traffic = [
                self.traffic_conditions[i, j] for j in range(self.n_locations) if i != j
            ]
            traffic_state[i] = np.mean(outgoing_traffic)
        
        return traffic_state
    
    def render(self, mode='human'):
        """
        Render the environment with traffic information.
        
        Args:
            mode (str): Rendering mode
        """
        # Call base render
        super(DynamicDeliveryEnvironment, self).render(mode)
        
        if mode == 'human':
            # Print traffic information
            traffic_info = self.traffic_simulator.get_traffic_info()
            print("\nTraffic Information:")
            print(f"Simulation time: {traffic_info['simulation_time']:.2f} hours")
            print(f"Hour of day: {traffic_info['hour_of_day']:.2f}")
            print(f"Active incidents: {traffic_info['n_active_incidents']}")
            print(f"Average traffic multiplier: {traffic_info['avg_traffic_multiplier']:.2f}")
            print(f"Max traffic multiplier: {traffic_info['max_traffic_multiplier']:.2f}")
    
    def get_traffic_heatmap(self):
        """
        Get a representation of traffic conditions for visualization.
        
        Returns:
            np.ndarray: Traffic heatmap
        """
        return self.traffic_conditions