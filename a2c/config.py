
import numpy as np
import tensorflow as tf



class Config(object):
    def __init__(self):
        self.debug = False
        self.game = 'Pong'

        self.env_versions = 'NoFrameskip-v4'

        self.tf_config = tf.ConfigProto()
        self.tf_config.gpu_options.allow_growth = True
        self.tf_dtype = tf.float32

        self.history_length = 4
        self.state_shape = [84, 84]
        self.state_dtype = np.float32

        self.no_op_max = 30
        self.action_repeat = 4
        self.discount = 0.99

        self.total_step = 10e6
        self.evaluate_every_step = 2e5

        self.evaluate_episodes = 30
        self.evaluate_episode_step = 5*60*15

        self.summary_dir='./summary/'
        self.model_dir='./model/'
		
        self.learning_rate = 7e-4
        self.lr_anneal = True
        self.max_grad_norm = 0.5

        self.nenvs = 8
        self.nsteps = 5


