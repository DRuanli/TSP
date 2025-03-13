import numpy as np
import time

class TrafficSimulator:
    """
    Simulates dynamic traffic conditions for the Delivery Route Optimization Problem.
    Generates time-dependent traffic patterns and random incidents.
    """
    
    def __init__(
        self,
        distance_matrix,
        base_traffic_conditions=None,
        time_factor=0.1,
        incident_probability=0.05,
        incident_severity=(1.5, 3.0),
        incident_duration=(5, 20),
        periodic_variation=True
    ):
        """
        Initialize the traffic simulator.
        
        Args:
            distance_matrix (np.ndarray): Base distance/travel time matrix
            base_traffic_conditions (np.ndarray, optional): Initial traffic conditions
            time_factor (float): How quickly time progresses in the simulation
            incident_probability (float): Probability of traffic incident per location per update
            incident_severity (tuple): Range of traffic multipliers for incidents (min, max)
            incident_duration (tuple): Range of incident durations in steps (min, max)
            periodic_variation (bool): Whether to include time-of-day traffic patterns
        """
        self.n_locations = distance_matrix.shape[0]
        self.distance_matrix = distance_matrix
        
        # Initialize base traffic conditions if not provided
        if base_traffic_conditions is None:
            self.base_traffic_conditions = np.ones_like(distance_matrix)
        else:
            self.base_traffic_conditions = base_traffic_conditions.copy()
            
        # Current traffic conditions
        self.current_traffic = self.base_traffic_conditions.copy()
        
        # Traffic simulation parameters
        self.time_factor = time_factor
        self.incident_probability = incident_probability
        self.incident_severity = incident_severity
        self.incident_duration = incident_duration
        self.periodic_variation = periodic_variation
        
        # Time tracking
        self.start_time = time.time()
        self.simulation_time = 0  # in hours
        
        # Active incidents
        self.active_incidents = {}  # (i, j): (severity, remaining_duration)
        
    def reset(self):
        """
        Reset the traffic conditions to base levels.
        
        Returns:
            np.ndarray: Reset traffic conditions
        """
        self.current_traffic = self.base_traffic_conditions.copy()
        self.start_time = time.time()
        self.simulation_time = 0
        self.active_incidents = {}
        return self.current_traffic
    
    def get_time_of_day_factor(self):
        """
        Calculate traffic multiplier based on time of day (24-hour cycle).
        
        Returns:
            np.ndarray: Matrix of time-of-day traffic multipliers
        """
        if not self.periodic_variation:
            return np.ones_like(self.distance_matrix)
        
        # Calculate hour of day (0-24)
        hour = (self.simulation_time % 24)
        
        # Morning rush hour (7-9 AM)
        if 7 <= hour < 9:
            base_factor = 1.5 - 0.5 * np.cos(np.pi * (hour - 7) / 2)
        # Evening rush hour (4-6 PM)
        elif 16 <= hour < 18:
            base_factor = 1.5 - 0.5 * np.cos(np.pi * (hour - 16) / 2)
        # Late night (11 PM - 5 AM)
        elif hour >= 23 or hour < 5:
            base_factor = 0.7
        # Normal daytime
        else:
            base_factor = 1.0
            
        # Create matrix with time factors
        time_factors = np.ones_like(self.distance_matrix) * base_factor
        
        # Diagonal elements (self-loops) should be 1.0
        np.fill_diagonal(time_factors, 1.0)
        
        return time_factors
    
    def generate_incidents(self):
        """
        Randomly generate new traffic incidents.
        """
        for i in range(self.n_locations):
            for j in range(self.n_locations):
                if i != j:
                    # Check if a new incident occurs
                    if np.random.random() < self.incident_probability:
                        # Generate incident severity
                        severity = np.random.uniform(
                            self.incident_severity[0],
                            self.incident_severity[1]
                        )
                        
                        # Generate incident duration
                        duration = np.random.randint(
                            self.incident_duration[0],
                            self.incident_duration[1] + 1
                        )
                        
                        # Add to active incidents
                        self.active_incidents[(i, j)] = (severity, duration)
    
    def update_incidents(self):
        """
        Update the status of active incidents.
        """
        # List to store incidents that have ended
        ended_incidents = []
        
        # Update each active incident
        for (i, j), (severity, duration) in self.active_incidents.items():
            # Reduce remaining duration
            duration -= 1
            
            # Check if incident has ended
            if duration <= 0:
                ended_incidents.append((i, j))
            else:
                # Update the incident
                self.active_incidents[(i, j)] = (severity, duration)
        
        # Remove ended incidents
        for incident in ended_incidents:
            del self.active_incidents[incident]
    
    def apply_incidents(self, traffic):
        """
        Apply active incidents to the traffic conditions.
        
        Args:
            traffic (np.ndarray): Base traffic conditions
            
        Returns:
            np.ndarray: Traffic conditions with incidents applied
        """
        # Create a copy to avoid modifying the input
        result = traffic.copy()
        
        # Apply each active incident
        for (i, j), (severity, _) in self.active_incidents.items():
            result[i, j] = severity
            
        return result
    
    def update(self, time_step=1.0):
        """
        Update traffic conditions based on time elapsed and random events.
        
        Args:
            time_step (float): Number of time steps to advance
            
        Returns:
            tuple: (new_traffic_conditions, traffic_change_magnitude)
        """
        # Store previous traffic for comparison
        previous_traffic = self.current_traffic.copy()
        
        # Update simulation time
        self.simulation_time += time_step * self.time_factor
        
        # Get time-of-day factors
        time_factors = self.get_time_of_day_factor()
        
        # Generate new incidents
        self.generate_incidents()
        
        # Update existing incidents
        self.update_incidents()
        
        # Apply time-of-day factors
        traffic = self.base_traffic_conditions * time_factors
        
        # Apply active incidents
        traffic = self.apply_incidents(traffic)
        
        # Update current traffic
        self.current_traffic = traffic
        
        # Calculate magnitude of change
        traffic_change = np.abs(self.current_traffic - previous_traffic)
        change_magnitude = np.mean(traffic_change)
        
        return self.current_traffic, change_magnitude
    
    def get_current_traffic(self):
        """
        Get the current traffic conditions.
        
        Returns:
            np.ndarray: Current traffic conditions
        """
        return self.current_traffic
    
    def get_traffic_info(self):
        """
        Get information about current traffic conditions.
        
        Returns:
            dict: Traffic information
        """
        return {
            "simulation_time": self.simulation_time,
            "hour_of_day": self.simulation_time % 24,
            "n_active_incidents": len(self.active_incidents),
            "incident_locations": list(self.active_incidents.keys()),
            "avg_traffic_multiplier": np.mean(self.current_traffic),
            "max_traffic_multiplier": np.max(self.current_traffic),
            "min_traffic_multiplier": np.min(self.current_traffic)
        }