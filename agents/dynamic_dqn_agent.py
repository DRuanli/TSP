import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.dqn_agent import DQNAgent
from utils.replay_buffer import PrioritizedReplayBuffer


class DynamicDQNAgent(DQNAgent):
    """
    Extended DQN Agent for handling dynamic traffic conditions.
    Adds features for adaptation to changing environment conditions.
    """
    
    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_dims=[128, 128],
        learning_rate=0.001,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        buffer_size=10000,
        batch_size=64,
        target_update_freq=10,
        prioritized_replay=True,
        alpha=0.6,
        beta=0.4,
        beta_increment=0.001,
        traffic_change_bonus=0.5,
        device="cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize the Dynamic DQN agent.
        
        Args:
            state_dim (int): Dimension of the state space
            action_dim (int): Dimension of the action space
            hidden_dims (list): Dimensions of hidden layers
            learning_rate (float): Learning rate for the optimizer
            gamma (float): Discount factor
            epsilon_start (float): Initial epsilon value for exploration
            epsilon_end (float): Final epsilon value for exploration
            epsilon_decay (float): Decay rate for epsilon
            buffer_size (int): Size of the replay buffer
            batch_size (int): Batch size for training
            target_update_freq (int): Frequency of target network updates
            prioritized_replay (bool): Whether to use prioritized replay
            alpha (float): How much prioritization to use (0 = uniform sampling)
            beta (float): Correction for importance sampling (0 = no correction)
            beta_increment (float): Increment of beta per sampling
            traffic_change_bonus (float): Bonus epsilon for traffic changes
            device (str): Device to use for training (cuda or cpu)
        """
        # Initialize the base DQN agent
        super(DynamicDQNAgent, self).__init__(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims,
            learning_rate=learning_rate,
            gamma=gamma,
            epsilon_start=epsilon_start,
            epsilon_end=epsilon_end,
            epsilon_decay=epsilon_decay,
            buffer_size=buffer_size,
            batch_size=batch_size,
            target_update_freq=target_update_freq,
            device=device
        )
        
        # Replace standard replay buffer with prioritized if specified
        if prioritized_replay:
            self.replay_buffer = PrioritizedReplayBuffer(
                capacity=buffer_size,
                alpha=alpha,
                beta=beta,
                beta_increment=beta_increment
            )
        
        # Additional parameters for dynamic traffic handling
        self.traffic_change_bonus = traffic_change_bonus
        self.prioritized_replay = prioritized_replay
        self.last_significant_update = False
    
    def select_action(self, state, valid_actions=None, traffic_changed=False):
        """
        Select an action using epsilon-greedy policy with traffic adaptation.
        
        Args:
            state (np.ndarray): Current state
            valid_actions (np.ndarray, optional): Boolean mask for valid actions
            traffic_changed (bool): Whether significant traffic change occurred
            
        Returns:
            int: Selected action
        """
        # Boost exploration on significant traffic changes
        current_epsilon = self.epsilon
        if traffic_changed and self.traffic_change_bonus > 0:
            # Temporarily increase epsilon for exploration
            current_epsilon = min(1.0, self.epsilon + self.traffic_change_bonus)
        
        # Convert state to tensor
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        # Create valid actions mask if provided
        if valid_actions is not None:
            valid_actions_tensor = torch.BoolTensor(valid_actions).to(self.device)
        else:
            valid_actions_tensor = None
        
        # Epsilon-greedy policy with dynamic epsilon
        if random.random() < current_epsilon:
            # Explore: select a random valid action
            if valid_actions is not None:
                valid_indices = np.where(valid_actions)[0]
                if len(valid_indices) > 0:
                    action = np.random.choice(valid_indices)
                else:
                    action = random.randrange(self.action_dim)
            else:
                action = random.randrange(self.action_dim)
        else:
            # Exploit: select the action with highest Q-value
            with torch.no_grad():
                q_values = self.policy_net.get_q_values(state_tensor, valid_actions_tensor)
                action = q_values.max(1)[1].item()
        
        # Decay epsilon (only use base epsilon)
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        # Update steps
        self.steps_done += 1
        
        # Store traffic change status
        self.last_significant_update = traffic_changed
        
        return action
    
    def store_transition(self, state, action, reward, next_state, done, traffic_changed=False):
        """
        Store a transition in the replay buffer with priority if enabled.
        
        Args:
            state (np.ndarray): Current state
            action (int): Action taken
            reward (float): Reward received
            next_state (np.ndarray): Next state
            done (bool): Whether the episode is done
            traffic_changed (bool): Whether significant traffic change occurred
        """
        if self.prioritized_replay:
            # Use prioritized replay - the buffer will handle the priorities internally
            super().store_transition(state, action, reward, next_state, done)
            
            # If we know this is a transition with traffic change, update its priority
            if traffic_changed and hasattr(self.replay_buffer, 'update_priorities'):
                # Use high priority for transitions with traffic changes
                # This is a simplification - we would need proper indices for real implementation
                # Just to illustrate the concept
                pass
        else:
            # Use standard replay
            super().store_transition(state, action, reward, next_state, done)
    
    def update(self):
        """
        Update the policy network using a batch of experiences.
        Extends the base DQN update with prioritized replay support.
        
        Returns:
            float: Loss value
        """
        # Check if we have enough samples
        if len(self.replay_buffer) < self.batch_size:
            return 0.0
        
        # Sample a batch of transitions (with priorities if enabled)
        if self.prioritized_replay and hasattr(self.replay_buffer, 'sample'):
            states, actions, rewards, next_states, dones, weights, indices = self.replay_buffer.sample(self.batch_size)
            # Convert weights to tensor
            weights = torch.FloatTensor(weights).to(self.device)
        else:
            states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
            weights = None
            indices = None
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Compute Q-values for current states and actions
        q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Compute Q-values for next states (using target network)
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            
        # Compute target Q-values
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Compute TD errors for prioritized replay
        td_errors = torch.abs(q_values - target_q_values).detach().cpu().numpy()
        
        # Compute weighted loss if using prioritized replay
        if weights is not None:
            # Apply importance sampling weights
            loss = torch.mean(weights * (q_values - target_q_values) ** 2)
        else:
            # Standard MSE loss
            loss = nn.MSELoss()(q_values, target_q_values)
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update priorities in buffer if using prioritized replay
        if self.prioritized_replay and indices is not None and hasattr(self.replay_buffer, 'update_priorities'):
            self.replay_buffer.update_priorities(indices, td_errors)
        
        # Update target network
        self.updates_done += 1
        if self.updates_done % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        return loss.item()
    
    def plan_route(self, env, start_location, unvisited):
        """
        Plan a complete route using the current policy.
        
        Args:
            env: Environment to plan in
            start_location (int): Starting location
            unvisited (list): List of unvisited locations
            
        Returns:
            list: Planned route
        """
        # Save environment state
        saved_state = env.reset()
        
        # Reset to custom state
        current_location = start_location
        visited = np.zeros(env.n_locations, dtype=bool)
        visited[current_location] = True
        for loc in unvisited:
            visited[loc] = False
        
        # Manually set environment state
        env.current_location = current_location
        env.visited = visited
        
        # Plan route
        route = [current_location]
        state = env._get_state()
        done = False
        
        while not done:
            valid_actions = env.get_valid_actions()
            action = self.select_action(state, valid_actions)
            next_state, _, done, info = env.step(action)
            
            if info["valid_action"]:
                route.append(action)
            
            state = next_state
        
        # Restore environment
        env.reset()
        
        return route
    
    def replan_if_needed(self, env, traffic_changed):
        """
        Decide whether to replan the route based on traffic changes.
        
        Args:
            env: Current environment
            traffic_changed (bool): Whether significant traffic change occurred
            
        Returns:
            bool: Whether replanning was performed
        """
        if not traffic_changed:
            return False
        
        # Get current state
        current_location = env.current_location
        unvisited = np.where(~env.visited)[0]
        
        if len(unvisited) == 0:
            return False  # No need to replan if all locations visited
        
        # Plan new route
        new_route = self.plan_route(env, current_location, unvisited)
        
        # In a real implementation, you might want to evaluate the new route vs. the old one
        # and decide whether to switch. For now, we always replan when traffic changes significantly.
        
        return True