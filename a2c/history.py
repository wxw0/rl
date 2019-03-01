
import numpy as np 

class History():
    def __init__(self, config):
        self.cf = config
        self.states = np.zeros([1] + self.cf.state_shape + [self.cf.history_length], dtype=self.cf.state_dtype)

    def add(self, state):
        self.states[..., :-1] = self.states[..., 1:]
        self.states[..., -1] = state

    def get(self):
        return self.states

    def clear(self):
        self.states[:] = 0

