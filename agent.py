import torch
import torch.nn.functional as F
import numpy as np
from model import QNetwork
from replay_buffer import ReplayBuffer
from utils import save_checkpoint

class DQNAgent:
    def __init__(self,
                 state_dim,
                 action_dim,
                 device='cpu',
                 lr=1e-3,
                 gamma=0.99,
                 batch_size=64,
                 buffer_size=100000,
                 epsilon_start=1.0,
                 epsilon_final=0.01,
                 epsilon_decay=500,
                 target_update=1000):
        self.device = torch.device(device)
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.epsilon_start = epsilon_start
        self.epsilon_final = epsilon_final
        self.epsilon_decay = epsilon_decay
        self.target_update = target_update

        self.policy_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr)
        self.replay = ReplayBuffer(buffer_size)

        self.steps_done = 0

    def select_action(self, state, eval_mode=False):
        # epsilon-greedy
        epsilon = self.epsilon_final + (self.epsilon_start - self.epsilon_final) * \
                  np.exp(-1.0 * self.steps_done / self.epsilon_decay)
        self.steps_done += 1
        if eval_mode or np.random.rand() > epsilon:
            state_t = torch.tensor(np.array(state, dtype=np.float32), device=self.device).unsqueeze(0)
            with torch.no_grad():
                qvals = self.policy_net(state_t)
                action = int(qvals.argmax().item())
            return action
        else:
            return np.random.randint(0, self.action_dim)

    def push_transition(self, s, a, r, s2, done):
        self.replay.push(s, a, r, s2, done)

    def update(self):
        if len(self.replay) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.replay.sample(self.batch_size)
        states_t = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions_t = torch.tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states_t = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        dones_t = torch.tensor(dones, dtype=torch.uint8, device=self.device).unsqueeze(1)

        q_values = self.policy_net(states_t).gather(1, actions_t)
        with torch.no_grad():
            next_q_values = self.target_net(next_states_t).max(1)[0].unsqueeze(1)
            target = rewards_t + self.gamma * next_q_values * (1 - dones_t.float())

        loss = F.mse_loss(q_values, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # soft/hard update target
        if self.steps_done % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return loss.item()

    def save(self, path):
        save_checkpoint(self.policy_net, path)

    def load(self, path, map_location=None):
        state = torch.load(path, map_location=map_location)
        self.policy_net.load_state_dict(state)
        self.target_net.load_state_dict(state)
