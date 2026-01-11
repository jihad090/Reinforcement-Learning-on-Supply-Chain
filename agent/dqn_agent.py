import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

    def forward(self, x):
        return self.net(x)

class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

        self.memory = deque(maxlen=5000)

        self.model = DQN(state_dim, action_dim)
        self.target_model = DQN(state_dim, action_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def act(self, state):
        if random.random() < self.epsilon:
            return random.randrange(5)
        state = torch.tensor(state, dtype=torch.float32)
        return torch.argmax(self.model(state)).item()

    def remember(self, s, a, r, ns, d):
        self.memory.append((s, a, r, ns, d))

    def replay(self, batch_size=64):
        if len(self.memory) < batch_size:
            return

        batch = random.sample(self.memory, batch_size)

        for state, action, reward, next_state, done in batch:

            state = torch.tensor(state, dtype=torch.float32)
            next_state = torch.tensor(next_state, dtype=torch.float32)

            reward = torch.tensor(reward, dtype=torch.float32)

            with torch.no_grad():
                next_q = self.target_model(next_state)
                max_next_q = torch.max(next_q)

            target = reward
            if not done:
                target = reward + self.gamma * max_next_q

            # ensure float32
            target = target.float()

            q_values = self.model(state)
            predicted_q = q_values[action]

            loss = nn.MSELoss()(predicted_q, target)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


    def update_target(self):
        self.target_model.load_state_dict(self.model.state_dict())
