# Delivery Route Optimization with Deep Q-Networks (DQN)

This project implements a solution for optimizing delivery routes using Deep Q-Networks (DQN), a reinforcement learning approach. The system can adapt to dynamic traffic conditions, making it valuable for real-world logistics applications.

## Overview

The delivery route optimization problem is a variant of the Traveling Salesman Problem (TSP) that incorporates real-time traffic data. The goal is to find the optimal route for visiting all delivery locations while minimizing travel time and adapting to changing traffic conditions.

This implementation includes:
- Standard DQN implementation for static environments
- Enhanced Dynamic DQN for handling traffic changes
- Simulation of traffic patterns and incidents
- Performance comparison tools for evaluating adaptation strategies

## Project Structure

```
.
├── agents/
│   ├── __init__.py
│   ├── dqn_agent.py                # Base DQN agent implementation
│   └── dynamic_dqn_agent.py        # Extended DQN for dynamic traffic 
├── environment/
│   ├── __init__.py
│   ├── delivery_env.py             # Base environment
│   └── dynamic_delivery_env.py     # Dynamic traffic environment
├── models/
│   ├── __init__.py
│   └── dqn_model.py                # Neural network architecture 
├── utils/
│   ├── __init__.py
│   ├── replay_buffer.py            # Experience replay implementation
│   ├── google_maps_api.py          # Google Maps integration (optional)
│   ├── traffic_simulator.py        # Traffic simulation tools
│   └── utils.py                    # Helper functions and visualization
├── main.py                         # Demo & testing script
├── train.py                        # Training for static DQN
├── train_dynamic.py                # Training for dynamic DQN
├── evaluate.py                     # Evaluation for static agent
├── evaluate_dynamic.py             # Evaluation for dynamic agent
├── compare_performance.py          # Comparative analysis tools
└── README.md                       # This documentation
```

## Features

### Basic Implementation (Task 1)
- Problem formulation for DQN in routing context
- Environment design with state space, action space, rewards
- DQN agent with neural network architecture
- Experience replay for stable learning
- Epsilon-greedy exploration

### Static Route Optimization (Task 2)
- Q-learning update rule implementation
- Google Maps API integration (optional)
- Visualization of routes and learning curves
- Performance tracking and evaluation tools

### Dynamic Traffic Adaptation (Task 3)
- Real-time traffic updates simulation
- Dynamic re-routing capabilities
- Prioritized experience replay
- Traffic visualization and change detection
- Performance comparison between static and dynamic agents

## Requirements

- Python 3.6+
- PyTorch
- NumPy
- Matplotlib
- NetworkX
- Gym (OpenAI Gym)
- Google Maps API key (optional, for real traffic data)

## Usage

### Training a Standard DQN Agent

```bash
python train.py --num_locations 5 --num_episodes 1000 --save_path trained_models/dqn_agent.pth
```

### Training a Dynamic DQN Agent

```bash
python train_dynamic.py --num_locations 5 --num_episodes 1000 --traffic_update_freq 5 --prioritized_replay --include_traffic_in_state
```

### Evaluating Agents

```bash
# Evaluate static agent
python evaluate.py --agent_path trained_models/dqn_agent.pth --num_episodes 10 --render

# Evaluate dynamic agent
python evaluate_dynamic.py --agent_path trained_models/dynamic_dqn_agent.pth --num_episodes 10 --render --traffic_intensity 0.5
```

### Comparing Performance

```bash
python compare_performance.py --static_agent_path trained_models/dqn_agent.pth --dynamic_agent_path trained_models/dynamic_dqn_agent.pth --num_episodes 5
```

## Key Components

### DQN Agent

The base DQN agent utilizes:
- Policy and target networks to stabilize learning
- Experience replay buffer to break correlated samples
- Epsilon-greedy exploration strategy
- Q-value approximation via neural networks

### Dynamic DQN Agent

Extends the base agent with:
- Prioritized experience replay for more efficient learning
- Traffic change detection and response
- Adaptive exploration based on traffic changes
- Route replanning capabilities

### Environment Design

- **State Space**: Current location (one-hot), visited status, traffic information
- **Action Space**: Next location to visit
- **Reward Function**: Negative travel time/distance
- **Termination**: All locations visited

### Traffic Simulation

- Time-of-day traffic patterns
- Random traffic incidents
- Configurable severity and duration
- Visualization tools for traffic changes

## Performance Evaluation

The system provides tools to compare:
- Static vs. dynamic agent performance
- Route quality across different traffic scenarios
- Adaptation to unexpected traffic changes
- Overall travel time and distance metrics

## Extensions

Possible extensions to this project:
- Multi-vehicle routing coordination
- Time windows for deliveries
- Capacity constraints
- Integration with real-time traffic APIs
- Deep Q-learning variants (Double DQN, Dueling DQN)

## License

[MIT License](LICENSE)