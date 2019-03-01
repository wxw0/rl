
import gym
import random
import time
import numpy as np
import cv2

from threading import Thread
from multiprocessing import Process, Pipe
from queue import Queue

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
                # if self.game == 'Pong' and r == -1:
                #         d = True
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



def env_work_thread(remote, par_remote, config):
    env = Env(config)
    try:
        while True:
            cmd, data = par_remote.get()
            if cmd == 'reset':
                env.reset()
                remote.put_nowait(env.recent_states[0])
            elif cmd == 'act':
                s, r, d = env.act(data)
                real_done = env.real_done
                if d:
                    env.reset()
                remote.put_nowait((env.recent_states[0], r, d, real_done))
            elif cmd == 'close':
                break
            elif cmd == 'render':
                env.env.render()
            else:
                raise NotImplementedError
    except Exception as e:
        print(e)
    finally:
        env.close()

class MultiEnv_thread(object):
    def __init__(self, config):
        print('MultiEnv_thread')
        self.cf = config
        nenvs = self.cf.nenvs
        self.remotes, self.work_remotes = zip(*[[Queue(1), Queue(1)] for i in range(nenvs)])
        self.ps = [Thread(target=env_work_thread, args=(wr, r, self.cf)) \
                    for (wr, r) in zip(self.work_remotes, self.remotes)]
        for p in self.ps:
            p.daemon = True
            p.start()
            time.sleep(1)

    def reset(self):
        for r in self.remotes:
            r.put_nowait(('reset', None))
        results = [remote.get() for remote in self.work_remotes]
        s = results
        return np.stack(s)

    def act(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.put_nowait(('act', action))
        results = [remote.get() for remote in self.work_remotes]
        s, r, d, real_done = zip(*results)
        return np.stack(s), np.stack(r), np.stack(d), np.stack(real_done)

    def render(self):
        for r in self.remotes: 
            r.put_nowait(('render', None))

    def close(self):
        for r in self.remotes: 
            r.put_nowait(('close', None))
        for p in self.ps:
            p.join()


