
import gym
import random
import numpy as np
import cv2


from history import History


class Env(object):
    def __init__(self, config):
        self.cf = config
        self.game = self.cf.game
        self.history = History(self.cf)
        self.env = gym.make(self.game + self.cf.env_versions)
        self.action_n = self.env.action_space.n
        self._obs = np.zeros((2,) + self.env.observation_space.shape, dtype=np.uint8)
        self.real_done = True
        print('env: ', self.game, self.action_n)

    def reset(self):
        self.history.clear()
        self._obs[:] = 0
        if not self.real_done:
            s, r, d = self.act(0)
        else:
            obs = self.env.reset()
            self._obs_add(obs)
            no_op = random.randint(1, self.cf.no_op_max - 1)
            s, r, d = self.act(0, no_op)
        if d:
            return self.reset()
        for i in range(self.cf.history_length):
            self.history.add(s)
        return s

    def act(self, action, action_repeat=None, is_training=True):
        if action_repeat == None:
            action_repeat = self.cf.action_repeat
        start_lives = self.lives
        total_r = 0
        for i in range(action_repeat):
            obs, r, d, info = self.env.step(action)
            s = self._obs_add(obs)
            self.real_done = d
            total_r += r
            
            if is_training:
                if self.lives < start_lives:
                        d = True
                if  d:
                    total_r = -1
            if d:
                break

        self.history.add(s)
        return s, total_r, d

    def sample_action(self):
        return self.env.action_space.sample()

    def close(self):
        return self.env.close()

    def _obs_add(self, obs):
        self._obs[0] = self._obs[1]
        self._obs[1] = obs
        state = np.max(self._obs, axis=0)

        state = cv2.resize(cv2.cvtColor(state, cv2.COLOR_RGB2GRAY), tuple(self.cf.state_shape), interpolation=cv2.INTER_AREA)
        return (state/255.).astype(self.cf.state_dtype)

    @property
    def lives(self):
        return self.env.env.ale.lives()

    @property
    def recent_states(self):
        return self.history.get()

