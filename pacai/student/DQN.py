import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque


class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )
    
    def forward(self, x):
        return self.fc(x)


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def add(self, experience):
        self.buffer.append(experience)
    
    def sample(self, batch_size):
        samples = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*samples)
        return (
            torch.tensor(states, dtype=torch.float32),
            torch.tensor(actions, dtype=torch.int64),
            torch.tensor(rewards, dtype=torch.float32),
            torch.tensor(next_states, dtype=torch.float32),
            torch.tensor(dones, dtype=torch.float32),
        )
    
    def size(self):
        return len(self.buffer)


def train_dqn():
    # Initialize environment
    state_dim = # 
    action_dim = env.action_space.n
    
    # Initialize DQN and target DQN
    dqn = DQN(state_dim, action_dim)
    target_dqn = DQN(state_dim, action_dim)
    target_dqn.load_state_dict(dqn.state_dict())
    target_dqn.eval()
    
    # Optimizer and loss
    optimizer = optim.Adam(dqn.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    
    # Replay buffer
    buffer = ReplayBuffer(capacity=10000)
    
    # Exploration parameters
    epsilon = 1.0
    epsilon_min = 0.01
    
    # Training loop
    rewards_history = []
    for episode in range(episodes):
        state = env.reset()[0]  # Gym 0.26+ returns (state, info)
        total_reward = 0
        
        while True:
            # Epsilon-greedy action selection
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    q_values = dqn(torch.tensor(state, dtype=torch.float32))
                    action = torch.argmax(q_values).item()
            
            # Take action in the environment
            next_state, reward, done, _, _ = env.step(action)
            total_reward += reward
            
            # Store experience in the replay buffer
            buffer.add((state, action, reward, next_state, done))
            state = next_state
            
            # Train if buffer is sufficiently full
            if buffer.size() >= batch_size:
                # Sample a batch
                states, actions, rewards, next_states, dones = buffer.sample(batch_size)
                
                # Compute target Q-values
                with torch.no_grad():
                    target_q = rewards + gamma * (1 - dones) * torch.max(target_dqn(next_states), dim=1)[0]
                
                # Compute current Q-values
                current_q = dqn(states).gather(1, actions.unsqueeze(1)).squeeze(1)
                
                # Compute loss
                loss = loss_fn(current_q, target_q)
                
                # Optimize the model
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            if done:
                break
        
        # Decay epsilon
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        
        # Update target network periodically
        if episode % 10 == 0:
            target_dqn.load_state_dict(dqn.state_dict())
        
        rewards_history.append(total_reward)
        print(f"Episode {episode}, Total Reward: {total_reward}, Epsilon: {epsilon:.3f}")
    
    env.close()
    return rewards_history
