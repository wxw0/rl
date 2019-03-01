
import os
import numpy as np
import random
import time

import tensorflow as tf
import tensorflow.contrib.slim as slim

from memory import Memory
from env import Env


def huber_loss(x, delta=1.0):
    # print('huber_loss')
    return tf.where(tf.abs(x) < delta, tf.square(x) * 0.5, delta * (tf.abs(x) - 0.5 * delta), name='huber_loss')

class Qnet(object):
    def __init__(self, config, action_n, scope=None):
        self.cf = config

        with tf.variable_scope(scope):
            self.input =  tf.placeholder(shape=[None]+self.cf.state_shape+[4],dtype=self.cf.tf_dtype)
            self.conv1 = slim.conv2d(self.input, activation_fn=tf.nn.relu, num_outputs=32, kernel_size=[8,8], stride=[4,4], padding='VALID')
            self.conv2 = slim.conv2d(self.conv1, activation_fn=tf.nn.relu, num_outputs=64, kernel_size=[4,4], stride=[2,2], padding='VALID')
            self.conv3 = slim.conv2d(self.conv2, activation_fn=tf.nn.relu, num_outputs=64, kernel_size=[3,3], stride=[1,1], padding='VALID')
            # print('conv: ', self.input.shape, self.conv1.shape, self.conv2.shape, self.conv3.shape)

            self.convFlat = tf.reshape(self.conv3, [-1, 3136])
            self.fc = slim.fully_connected(self.convFlat, 512, activation_fn=tf.nn.relu)
            self.Qout = slim.fully_connected(self.fc, action_n, activation_fn=None)
            # print('Qout: ', self.convFlat.shape, self.fc.shape, self.Qout.shape)

            self.predict = tf.argmax(self.Qout, 1)
            # print('predict: ', self.predict.shape)

class mainQnet(Qnet):
    def __init__(self, config, action_n, scope=None):
        super(mainQnet, self).__init__(config, action_n=action_n, scope=scope)

        with tf.variable_scope('action_onehot'):
            self.action = tf.placeholder(shape=[None], dtype=tf.int32)
            self.action_onehot = tf.one_hot(self.action, action_n, dtype=self.cf.tf_dtype)

        with tf.variable_scope('Q'):
            self.Q = tf.reduce_sum(tf.multiply(self.Qout, self.action_onehot), axis=1, name='Q')
            # print('Q: ', self.action.shape, self.action_onehot.shape, self.Q.shape)

        self.targetQ = tf.placeholder(shape=[None], dtype=self.cf.tf_dtype, name='targetQ')

        with tf.variable_scope('q_loss'):
            self.td_error = tf.subtract(self.Q, self.targetQ, name='td_error')
            if self.cf.huber_loss:
                self.q_loss = tf.reduce_mean(huber_loss(self.td_error))
            else:
                self.q_loss = tf.reduce_mean(tf.square(self.td_error))

        with tf.variable_scope('loss'):
            self.loss = self.q_loss

        with tf.variable_scope('train'):
            params = tf.trainable_variables(scope)
            grads = tf.gradients(self.loss, params)
            grads, self.grad_norm = tf.clip_by_global_norm(grads, self.cf.max_grad_norm)
            grads = list(zip(grads, params))

            _trainer = tf.train.AdamOptimizer(learning_rate=self.cf.learning_rate)
            self.trainer = _trainer.apply_gradients(grads)


class DQN(object):
    def __init__(self, config, sess):
        self.cf = config
        self.sess = sess
        self.env = Env(self.cf)
        self.eval_env = Env(self.cf)

        self.mainQnet = mainQnet(self.cf, action_n=self.env.action_n, scope='mainQnet')
        self.targetQnet = Qnet(self.cf, action_n=self.env.action_n, scope='targetQnet')

        main_vars = tf.trainable_variables('mainQnet')
        target_vars = tf.trainable_variables('targetQnet')
        self.update_targetQnet_ops = []
        for v, tv in zip(main_vars, target_vars):
            self.update_targetQnet_ops.append(tv.assign(v))

        self.model_dir = self.cf.model_dir
        self.saver = tf.train.Saver(max_to_keep=1)


    def predict_a(self, state):
        net = self.mainQnet
        a, Qout = self.sess.run([net.predict, net.Qout], {net.input: state})
        # print('predict_a:', a, Qout)
        return a

    def train_mainQnet(self, step):
        pre_state, action, reward, done, post_state = self.memory.sample()
        targetQout = self.sess.run(self.targetQnet.Qout, {self.targetQnet.input: post_state})
        targetQmax = np.max(targetQout, axis=1)
        # print('targetQout:', targetQout, targetQout.shape)
        # print('targetQmax:', targetQmax, targetQmax.shape)

        # print('done: ', 1. - done)
        targetQ = (1. - done) * self.cf.discount * targetQmax + reward
        # print('targetQ: ', targetQ, targetQ.shape)

        net = self.mainQnet
        run_ops = [net.trainer, net.grad_norm, net.q_loss]
        results = self.sess.run(run_ops, {net.input: pre_state, net.action:action, net.targetQ: targetQ})

        # if results[1] > self.cf.max_grad_norm:
        # if True:
        #     print(*results[1:])

        for i in results[1:]:
            assert not np.isnan(i)
        self.mgn_avg.append(results[1])
        self.q_loss_avg.append(results[2])

    def update_targetQnet(self):
        # print('update_targetQnet...\n')
        self.sess.run(self.update_targetQnet_ops)

    def get_action(self, step):
        if step < self.cf.memory_start_size:
            return self.env.sample_action()

        if self.explore > self.cf.final_explore:
            self.explore -= self.explore_descend
        else:
            self.explore = self.cf.final_explore

        if random.random() > self.explore:
            action = self.predict_a(self.env.recent_states)[0]
            # print('predict_a:', action)
        else:
            action = self.env.sample_action()
            # print('sample_action:', action)
        return action

    def learn(self):
        self.memory = Memory(self.cf)
        self.explore = self.cf.init_explore
        self.explore_descend = (self.cf.init_explore - self.cf.final_explore) / self.cf.final_explore_step

        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        self.summary_dir = self.cf.summary_dir

        self.summary_write = tf.summary.FileWriter(self.summary_dir, self.sess.graph)
        self.summary_write.flush()

        self.episode_r_summary = tf.Variable(0., trainable=False)
        er_op = tf.summary.scalar('r/episode_r_avg', self.episode_r_summary)
        self.eval_episode_r_summary = tf.Variable(0., trainable=False)
        eer_op = tf.summary.scalar('r/evaluate_episode_r_avg', self.eval_episode_r_summary)
        self.q_loss_summary = tf.Variable(0., trainable=False)
        ql_op = tf.summary.scalar('loss/q_loss_avg', self.q_loss_summary)
        self.mgn_summary = tf.Variable(0., trainable=False)
        mgn_op = tf.summary.scalar('loss/mgn_avg', self.mgn_summary)

        self.summary_op = tf.summary.merge([er_op, eer_op, ql_op, mgn_op])

        self.q_loss_avg = []
        self.mgn_avg = []


        print('\nLearning...\n')
        self.update_targetQnet()
        step = 0
        state = self.env.reset()
        done = False
        episode = 1
        episode_step = 0
        episode_reward = 0
        episodes_average = []

        best_score = -9999.
        while step < self.cf.total_step:
            step += 1
            episode_step += 1

            action = self.get_action(step)
            state_1, reward, done = self.env.act(action)
            # print('reward: ', reward, done)
            self.memory.add(state, action, np.sign(reward), done)

            episode_reward += reward
            state = state_1
            if done:
                if self.env.real_done:
                    episodes_average.append(episode_reward)
                    episode += 1
                    episode_step = 0
                    episode_reward = 0
                # self.memory.add(state, 0, 0, False)
                state = self.env.reset()
                done = False

            if step > self.cf.memory_start_size:
                if step % self.cf.train_frequency == 0:
                    self.train_mainQnet(step)
                if step % self.cf.update_frequency == 0:
                    self.update_targetQnet()
                if step % self.cf.evaluate_every_step == 0:
                    
                    episode_r = np.array(episodes_average)
                    q_l_a = np.array(self.q_loss_avg)
                    mgn_a = np.array(self.mgn_avg)

                    eval_episode_r = self.evaluate()
                    summary_op = self.sess.run(self.summary_op, {self.episode_r_summary: episode_r.mean(), 
                                                self.eval_episode_r_summary: eval_episode_r.mean(), 
                                                self.mgn_summary: mgn_a.mean(), 
                                                self.q_loss_summary:q_l_a.mean()})
                    self.summary_write.add_summary(summary_op, global_step=step)
                    self.summary_write.flush()

                    episodes_average = []
                    self.q_loss_avg = []
                    self.mgn_avg = []

                    with open(self.summary_dir + 'r.csv', 'a') as f:
                        r_csv = str(time.time()) + ',' + str(step) + ',' + str(episode_r.mean()) + ',' + str(episode_r.std()) +\
                                ',' + str(eval_episode_r.mean()) + ',' + str(eval_episode_r.std()) +\
                                ',' + str(mgn_a.mean()) + ',' + str(mgn_a.std()) + \
                                ',' + str(q_l_a.mean()) + ',' + str(q_l_a.std()) + '\n'
                        print(r_csv)
                        f.write(r_csv)

                    if eval_episode_r.mean() > best_score:
                        best_score = eval_episode_r.mean()
                        self.saver.save(self.sess, self.model_dir + str(step))
        self.env.close()


    def get_action_for_evaluate(self):
        if random.random() > self.cf.evaluate_explore:
            action = self.predict_a(self.eval_env.recent_states)
        else:
            action = self.eval_env.sample_action()
        return action

    def evaluate(self, load_model = False):

        if load_model:
            print('\nEvaluate...')
            print('Loading Model...' + self.model_dir + '\n')
            ckpt_state = tf.train.get_checkpoint_state(self.model_dir)
            print('ckpt_state: ', ckpt_state.model_checkpoint_path)
            self.saver.restore(self.sess,ckpt_state.model_checkpoint_path)

        self.eval_env.reset()
        episode_step = 0
        episode_reward = 0
        episodes_average = []

        while len(episodes_average) < self.cf.evaluate_episodes:
            episode_step += 1
            action = self.get_action_for_evaluate()
            state, reward, done = self.eval_env.act(action, is_training=False)
            episode_reward += reward

            if done or episode_step > self.cf.evaluate_episode_step:
                # print('evaluate episode_step: ', episode_step)
                self.eval_env.real_done = True
                episodes_average.append(episode_reward)
                episode_step = 0
                episode_reward = 0
                self.eval_env.reset()

        e_a = np.array(episodes_average)
        # print('evaluate: ', 'episodes: ', e_a.size, 'average: ', e_a.mean(), 'std: ', e_a.std())
        return e_a

