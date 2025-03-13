import numpy as np
import gym
from gym import spaces
import networkx as nx


class DeliveryEnvironment(gym.Env):
    """
    Environment for the Delivery Route Optimization Problem using DQN.
    This environment simulates a delivery scenario where an agent needs to visit
    multiple locations in an optimal order, considering traffic conditions.
    """
    
    def __init__(self, locations, distance_matrix, traffic_conditions=None):
        """
        Initialize the Delivery Environment.
        
        Args:
            locations (list): List of location coordinates (e.g., [(lat1, lon1), (lat2, lon2), ...])
            distance_matrix (np.ndarray): Matrix of distances/travel times between locations
            traffic_conditions (np.ndarray, optional): Matrix of traffic multipliers (1.0 = normal, >1.0 = congestion)
        """
        super(DeliveryEnvironment, self).__init__()
        
        self.locations = locations
        self.n_locations = len(locations)
        self.distance_matrix = distance_matrix
        
        # Initialize traffic conditions (default: no traffic)
        if traffic_conditions is None:
            self.traffic_conditions = np.ones_like(distance_matrix)
        else:
            self.traffic_conditions = traffic_conditions
        
        # State space: current location (one-hot) + visited status for each location (binary)
        # Shape: n_locations (one-hot) + n_locations (binary visited status)
        self.observation_space = spaces.Box(
            low=0, 
            high=1,
            shape=(self.n_locations * 2,),
            dtype=np.float32
        )
        
        # Action space: choose the next location to visit (0 to n_locations-1)
        self.action_space = spaces.Discrete(self.n_locations)
        
        # Initialize state variables
        self.current_location = None
        self.visited = None
        self.steps = None
        self.total_distance = None
        
    def reset(self):
        """
        Reset the environment to start a new episode.
        Start at the depot (location 0) and mark all locations as unvisited.
        
        Returns:
            np.ndarray: Initial state representation
        """
        self.current_location = 0  # Start at the depot
        self.visited = np.zeros(self.n_locations, dtype=bool)
        self.visited[0] = True  # Mark depot as visited
        self.steps = 0
        self.total_distance = 0.0
        
        return self._get_state()
    
    def step(self, action):
        """
        Take an action (visit a location) and return the new state, reward, done flag, and info.
        
        Args:
            action (int): The location to visit next (0 to n_locations-1)
            
        Returns:
            tuple: (next_state, reward, done, info)
        """
        # Check if the action is valid (location not visited yet)
        if self.visited[action] or action == self.current_location:
            # Invalid action (location already visited or staying at the same location)
            reward = -10.0  # Penalty for invalid action
            done = False
            info = {"valid_action": False}
            return self._get_state(), reward, done, info
        
        # Calculate travel distance/time with traffic conditions
        travel_time = self.distance_matrix[self.current_location, action] * \
                     self.traffic_conditions[self.current_location, action]
        
        # Update state
        prev_location = self.current_location
        self.current_location = action
        self.visited[action] = True
        self.steps += 1
        self.total_distance += travel_time
        
        # Calculate reward (negative travel time)
        reward = -travel_time
        
        # Check if all locations have been visited
        all_visited = np.all(self.visited)
        
        # If all locations visited, add final return to depot (if not already there)
        if all_visited and self.current_location != 0:
            final_return = self.distance_matrix[self.current_location, 0] * \
                          self.traffic_conditions[self.current_location, 0]
            self.total_distance += final_return
            reward -= final_return  # Additional reward (negative) for returning to depot
            self.current_location = 0  # Return to depot
        
        # Check termination conditions
        done = all_visited
        
        # Additional information
        info = {
            "valid_action": True,
            "travel_time": travel_time,
            "total_distance": self.total_distance,
            "from_location": prev_location,
            "to_location": action,
            "steps": self.steps
        }
        
        return self._get_state(), reward, done, info
    
    def _get_state(self):
        """
        Convert the current environment state into a vector representation.
        
        Returns:
            np.ndarray: State representation combining current location (one-hot) and visited status
        """
        # One-hot encoding of current location
        current_loc_onehot = np.zeros(self.n_locations)
        current_loc_onehot[self.current_location] = 1
        
        # Visited status (binary vector)
        visited_status = self.visited.astype(np.float32)
        
        # Combine into state vector
        state = np.concatenate([current_loc_onehot, visited_status])
        
        return state
    
    def get_valid_actions(self):
        """
        Get a mask of valid actions (unvisited locations).
        
        Returns:
            np.ndarray: Boolean mask where True indicates a valid (unvisited) location
        """
        # Valid actions are locations that haven't been visited yet
        valid_actions = ~self.visited
        
        return valid_actions
    
    def render(self, mode='human'):
        """
        Render the environment (for visualization).
        
        Args:
            mode (str): Rendering mode
        """
        if mode == 'human':
            # Print current state
            print(f"Current location: {self.current_location}")
            print(f"Visited locations: {np.where(self.visited)[0]}")
            print(f"Unvisited locations: {np.where(~self.visited)[0]}")
            print(f"Total distance/time: {self.total_distance:.2f}")
            print(f"Steps: {self.steps}")