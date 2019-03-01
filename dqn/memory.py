
import numpy as np 
import random


class Memory():
    def __init__(self, config):
        self.cf = config
        self.size = int(self.cf.memory_size)
        self.state = np.empty(self.cf.state_shape + [self.size], dtype=self.cf.state_dtype)
        self.action = np.empty(self.size, dtype=np.int)
        self.reward = np.empty(self.size, dtype=np.float32)
        self.done = np.empty(self.size, dtype=np.bool)

        self.count = 0
        self.current_index = 0

        self.pre_state = np.empty([self.cf.batch_size] + self.cf.state_shape + [self.cf.history_length], dtype=self.cf.state_dtype)
        self.post_state = np.empty([self.cf.batch_size] + self.cf.state_shape + [self.cf.history_length], dtype=self.cf.state_dtype)

    def add(self, state, action, reward, done):
        self.state[..., self.current_index] = state
        self.action[self.current_index] = action
        self.reward[self.current_index] = reward
        self.done[self.current_index] = done

        self.count = self.count + 1 if self.count < self.size else self.size
        self.current_index = (self.current_index + 1) % self.size


    def get_state(self, index):
        return self.state[..., index - self.cf.history_length + 1 : index + 1]
        # return np.transpose(self.state[index - self.cf.history_length + 1 : index + 1], (1, 2, 0))

    def sample_index(self, batch_size):
        indexs = []
        while len(indexs) < batch_size:
            index = random.randint(self.cf.history_length, self.count - 2)
            if self.done[index - self.cf.history_length + 1: index].any():
                # print('done: ', index, self.state[0, 0, index])
                continue
            if self.current_index - 1 <= index and index <= self.current_index + self.cf.history_length - 2:
            # if index - self.cf.history_length + 2 <= self.current_index and self.current_index <= index + 1:
                # print('current_index: ', index, self.current_index, self.state[0, 0, index])
                continue
            indexs.append(index)
        # print('sample_index: ', indexs)
        # print(self.state[0, 0, indexs])
        return indexs

    def sample(self, batch_size=None):
        if batch_size == None:
            batch_size = self.cf.batch_size
        indexs = self.sample_index(batch_size)
        for i, index in enumerate(indexs):
            self.pre_state[i] = self.get_state(index)
            self.post_state[i] = self.get_state(index + 1)

        action = self.action[indexs]
        reward = self.reward[indexs]
        done = self.done[indexs]
        self.post_state[done] = 0

        return self.pre_state, action, reward, done, self.post_state


