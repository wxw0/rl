
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


        self.batch_size = 32
        self.history_length = 4
        self.state_shape = [84, 84]
        self.state_dtype = np.float16


        self.no_op_max = 30
        self.action_repeat = 4


        self.discount = 0.99
        self.learning_rate = 0.0001
        self.train_frequency = 4
        self.update_frequency = 1e4
        self.huber_loss = True
        self.max_grad_norm = 0.5


        self.init_explore = 1.
        self.final_explore = 0.05
        self.final_explore_step = 1e6


        self.memory_size = 2e5
        self.memory_start_size = 5e4

        self.total_step = 1e7

        self.evaluate_every_step = 2e5
        self.evaluate_episodes = 30
        self.evaluate_explore = 0.05
        self.evaluate_episode_step = 5*60*15

        self.summary_dir='./summary/'
        self.model_dir='./model/'



        
