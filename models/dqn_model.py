import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.dqn_model import DQNModel
from utils.replay_buffer import ReplayBuffer


class DQNAgent:
    """
    DQN Agent for the Delivery Route Optimization Problem.
    Uses a Deep Q-Network to learn the optimal policy.
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
        device="cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize the DQN agent.
        
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
            device (str): Device to use for training (cuda or cpu)
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.device = device
        self.learning_rate = learning_rate
        
        # Initialize Q-Networks (policy and target)
        self.policy_net = DQNModel(state_dim, action_dim, hidden_dims).to(device)
        self.target_net = DQNModel(state_dim, action_dim, hidden_dims).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Target network is used for evaluation only
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        
        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size)
        
        # Initialize training parameters
        self.steps_done = 0
        self.updates_done = 0
    
    def select_action(self, state, valid_actions=None):
        """
        Select an action using epsilon-greedy policy.
        
        Args:
            state (np.ndarray): Current state
            valid_actions (np.ndarray, optional): Boolean mask for valid actions
            
        Returns:
            int: Selected action
        """
        # Convert state to tensor
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        # Create valid actions mask if provided
        if valid_actions is not None:
            valid_actions_tensor = torch.BoolTensor(valid_actions).to(self.device)
        else:
            valid_actions_tensor = None
        
        # Epsilon-greedy policy
        if random.random() < self.epsilon:
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
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        # Update steps
        self.steps_done += 1
        
        return action
    
    def store_transition(self, state, action, reward, next_state, done):
        """
        Store a transition in the replay buffer.
        
        Args:
            state (np.ndarray): Current state
            action (int): Action taken
            reward (float): Reward received
            next_state (np.ndarray): Next state
            done (bool): Whether the episode is done
        """
        self.replay_buffer.add(state, action, reward, next_state, done)
    
    def update(self):
        """
        Update the policy network using a batch of experiences.
        
        Returns:
            float: Loss value
        """
        # Check if we have enough samples
        if len(self.replay_buffer) < self.batch_size:
            return 0.0
        
        # Sample a batch of transitions
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
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
        
        # Compute loss (MSE)
        loss = nn.MSELoss()(q_values, target_q_values)
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update target network
        self.updates_done += 1
        if self.updates_done % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        return loss.item()
    
    def save(self, path):
        """
        Save the agent's policy network.
        
        Args:
            path (str): Path to save the model
        """
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps_done': self.steps_done,
            'updates_done': self.updates_done
        }, path)
    
    def load(self, path):
        """
        Load the agent's policy network.
        
        Args:
            path (str): Path to load the model
        """
        checkpoint = torch.load(path)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.steps_done = checkpoint['steps_done']
        self.updates_done = checkpoint['updates_done']