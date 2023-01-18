import numpy as np
import collections

class PrioritizedReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.alpha = 0.5
        self.beta = 0.5

    def add(self, transition, priority=None):
        if priority is None:
            priority = self.priorities.max() if self.memory else 1.0
        if len(self.memory) < self.capacity:
            self.memory.append(transition)
        else:
            self.memory[self.position] = transition
        self.priorities[self.position] = priority
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        if len(self.memory) == self.capacity:
            priorities = self.priorities
        else:
            priorities = self.priorities[:self.position]
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()
        indices = np.random.choice(len(self.memory), batch_size, p=probabilities)
        transitions = [self.memory[i] for i in indices]
        weights = (len(self.memory) * probabilities[indices]) ** (-beta)
        weights /= weights.max()
        return transitions, indices, weights

    def update(self, index, priority):
        self.priorities[index.astype(np.int32)] = priority
