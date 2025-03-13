import numpy as np
import random
from collections import deque


class ReplayBuffer:
    """
    Experience Replay Buffer for DQN.
    Stores and samples transitions (state, action, reward, next_state, done).
    """
    
    def __init__(self, capacity):
        """
        Initialize the replay buffer with a fixed capacity.
        
        Args:
            capacity (int): Maximum number of transitions to store
        """
        self.buffer = deque(maxlen=capacity)
        
    def add(self, state, action, reward, next_state, done):
        """
        Add a new transition to the buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether the episode is done
        """
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """
        Sample a batch of transitions from the buffer.
        
        Args:
            batch_size (int): Number of transitions to sample
            
        Returns:
            tuple: Batch of (states, actions, rewards, next_states, dones)
        """
        # Make sure we have enough samples
        batch_size = min(batch_size, len(self.buffer))
        
        # Sample random transitions
        batch = random.sample(self.buffer, batch_size)
        
        # Separate the components
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to numpy arrays
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        dones = np.array(dones)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        """
        Get the current size of the buffer.
        
        Returns:
            int: Number of transitions in the buffer
        """
        return len(self.buffer)


class PrioritizedReplayBuffer(ReplayBuffer):
    """
    Prioritized Experience Replay Buffer for DQN.
    Samples transitions with priority based on TD error.
    """
    
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment=0.001, epsilon=0.01):
        """
        Initialize the prioritized replay buffer.
        
        Args:
            capacity (int): Maximum number of transitions to store
            alpha (float): How much prioritization to use (0 = uniform sampling)
            beta (float): Correction for importance sampling (0 = no correction)
            beta_increment (float): Increment of beta per sampling
            epsilon (float): Small constant to add to priorities
        """
        super(PrioritizedReplayBuffer, self).__init__(capacity)
        self.priorities = deque(maxlen=capacity)
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = epsilon
        self.max_priority = 1.0
        
    def add(self, state, action, reward, next_state, done):
        """
        Add a new transition to the buffer with max priority.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether the episode is done
        """
        # Add transition to the buffer
        super().add(state, action, reward, next_state, done)
        
        # Add with max priority (ensures it will be sampled at least once)
        self.priorities.append(self.max_priority)
    
    def sample(self, batch_size):
        """
        Sample a batch of transitions based on their priorities.
        
        Args:
            batch_size (int): Number of transitions to sample
            
        Returns:
            tuple: (states, actions, rewards, next_states, dones, weights, indices)
        """
        # Make sure we have enough samples
        batch_size = min(batch_size, len(self.buffer))
        
        # Convert priorities to sampling probabilities
        priorities = np.array(self.priorities)
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()
        
        # Sample indices based on probabilities
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        
        # Get the transitions
        batch = [self.buffer[i] for i in indices]
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Calculate importance sampling weights
        weights = (len(self.buffer) * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()  # Normalize weights
        
        # Increase beta
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        # Convert to numpy arrays
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        dones = np.array(dones)
        weights = np.array(weights)
        
        return states, actions, rewards, next_states, dones, weights, indices
    
    def update_priorities(self, indices, td_errors):
        """
        Update the priorities of the transitions.
        
        Args:
            indices (list): Indices of the transitions to update
            td_errors (list): TD errors of the transitions
        """
        for i, td_error in zip(indices, td_errors):
            # Calculate priority (add epsilon to avoid zero priority)
            priority = (abs(td_error) + self.epsilon) ** self.alpha
            
            # Update priority in the buffer
            self.priorities[i] = priority
            
            # Update max priority
            self.max_priority = max(self.max_priority, priority)