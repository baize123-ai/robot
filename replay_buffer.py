import random
import numpy as np
from collections import deque

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = int(capacity)
        self.buffer = deque(maxlen=self.capacity)

    def push(self, state, action, reward, next_state, done):
        # store as numpy arrays for efficiency
        self.buffer.append((np.array(state, dtype=np.float32),
                            int(action),
                            float(reward),
                            np.array(next_state, dtype=np.float32),
                            bool(done)))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.stack(states),
                np.array(actions, dtype=np.int64),
                np.array(rewards, dtype=np.float32),
                np.stack(next_states),
                np.array(dones, dtype=np.uint8))

    def __len__(self):
        return len(self.buffer)
