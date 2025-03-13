import torch
import torch.nn as nn
import torch.nn.functional as F


class DQNModel(nn.Module):
    """
    Neural network architecture for the DQN agent.
    Maps state representations to Q-values for each action.
    """
    
    def __init__(self, input_dim, output_dim, hidden_dims=[128, 128]):
        """
        Initialize the DQN model.
        
        Args:
            input_dim (int): Dimension of the state space
            output_dim (int): Dimension of the action space (number of locations)
            hidden_dims (list): Dimensions of hidden layers
        """
        super(DQNModel, self).__init__()
        
        # Create layers with the specified dimensions
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        # Output layer (Q-values for each action)
        layers.append(nn.Linear(prev_dim, output_dim))
        
        # Combine all layers
        self.layers = nn.Sequential(*layers)
        
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input state
            
        Returns:
            torch.Tensor: Q-values for each action
        """
        return self.layers(x)
    
    def get_q_values(self, state, valid_actions_mask=None):
        """
        Get Q-values for a given state, optionally masking invalid actions.
        
        Args:
            state (torch.Tensor): Input state
            valid_actions_mask (torch.Tensor, optional): Boolean mask for valid actions
            
        Returns:
            torch.Tensor: Q-values (masked if valid_actions_mask is provided)
        """
        q_values = self.forward(state)
        
        if valid_actions_mask is not None:
            # Make sure the mask has the same dimensions as q_values
            if q_values.dim() != valid_actions_mask.dim():
                valid_actions_mask = valid_actions_mask.unsqueeze(0)
            
            # Set Q-values of invalid actions to a very negative value
            invalid_mask = ~valid_actions_mask
            q_values = q_values.masked_fill(invalid_mask, float('-inf'))
        
        return q_values