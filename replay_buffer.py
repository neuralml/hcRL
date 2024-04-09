import random

import numpy as np
import torch

from collections import namedtuple

Transition = namedtuple("Transition", ("state", "action", "reward", "next_state"))


def make_replay(agent, capacity, sequence_length, device):
    if agent == "drqn":
        replay_memory = RecurrentExperienceReplayMemory(capacity, sequence_length)
    else:
        replay_memory = ReplayMemory(capacity, device)

    return replay_memory


class ReplayMemory(object):
    def __init__(self, capacity, device):
        self.capacity = capacity
        self.device = device
        self.memory = []
        self.position = 0
        self.counter = 0

    # Save a transition
    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)

        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity
        self.counter += 1

    def reset(self):
        self.memory = []
        self.position = 0
        self.counter = 0

    def sample(self, batch_size):
        if len(self.memory) < batch_size:
            batch_size = len(self.memory)
        transitions = random.sample(self.memory, batch_size)
        batch = Transition(*zip(*transitions))
        state = torch.as_tensor(batch.state, device=self.device).float()
        action = torch.as_tensor(batch.action, device=self.device).unsqueeze(-1)
        reward = torch.as_tensor(batch.reward, device=self.device).float().unsqueeze(-1)
        next_state = torch.as_tensor(batch.next_state, device=self.device).float()
        return state, action, reward, next_state

    def __len__(self):
        return len(self.memory)


class RecurrentExperienceReplayMemory:
    def __init__(self, capacity, sequence_length):
        self.capacity = capacity
        self.memory = []
        self.seq_length = sequence_length

    def push(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))
        if len(self.memory) > self.capacity:
            del self.memory[0]

    def reset(self):
        self.memory = []

    def sample(self, batch_size):
        finish = random.sample(
            range(len(self.memory) - 1, len(self.memory)), batch_size
        )
        begin = [x - self.seq_length for x in finish]

        samp = []
        for start, end in zip(begin, finish):
            # correct for sampling near beginning
            final = self.memory[max(start + 1, 0) : end + 1]

            # correct for sampling across episodes
            for i in range(len(final) - 2, -1, -1):
                if final[i][3] is None:
                    final = final[i + 1 :]
                    break

            # pad beginning to account for corrections
            while len(final) < self.seq_length:
                final = [
                    (
                        np.zeros_like(self.memory[0][0]),
                        0,
                        0,
                        np.zeros_like(self.memory[0][3]),
                    )
                ] + final

            samp += final

        # returns flattened version
        return samp, None, None

    def __len__(self):
        return len(self.memory)
