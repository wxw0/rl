
import os
import sys
import numpy as np
import random
import time

import tensorflow as tf
import tensorflow.contrib.slim as slim

from env import Env, MultiEnv_thread


class ACnet(object):
    def __init__(self, config, action_n):
        self.cf = config

        self.input =  tf.placeholder(shape=[None]+self.cf.state_shape+[self.cf.history_length],dtype=self.cf.tf_dtype)

        with tf.variable_scope('net_params'):
            self.conv1 = slim.conv2d(self.input, num_outputs=32, kernel_size=[8,8], stride=[4,4], padding='VALID')
            self.conv2 = slim.conv2d(self.conv1, num_outputs=64, kernel_size=[4,4], stride=[2,2], padding='VALID')
            self.conv3 = slim.conv2d(self.conv2, num_outputs=64, kernel_size=[3,3], stride=[1,1], padding='VALID')

        with tf.variable_scope('fc_params'):
            self.convFlat = tf.reshape(self.conv3, [-1, 3136])
            self.fc_concat = slim.fully_connected(self.convFlat, 512)

            self.fc = self.fc_concat

        self.logits = slim.fully_connected(self.fc, action_n, activation_fn=None, scope='logits_params')
        self.value = slim.fully_connected(self.fc, 1, activation_fn=None, scope='value_params')

        self.policy = slim.softmax(self.logits, scope='policy')

        with tf.variable_scope('action_onehot'):
            self.action = tf.placeholder(shape=[None], dtype=tf.int32)
            self.action_onehot = tf.one_hot(self.action, action_n, dtype=self.cf.tf_dtype)

        self.R = tf.placeholder(shape=[None], dtype=self.cf.tf_dtype, name='R')
        self.advantages = tf.placeholder(shape=[None], dtype=self.cf.tf_dtype, name='advantages')

        self.policy_action = tf.multinomial(self.logits, 1, name='sample_action')

        self.log_policy = tf.log(tf.clip_by_value(self.policy, 0.000001, 0.999999), name='log_policy')

        with tf.variable_scope('policy_loss'):
            self.log_policy_action = tf.reduce_sum(self.action_onehot * self.log_policy, axis=-1)
            self.policy_loss = - tf.reduce_mean(self.log_policy_action * self.advantages)

        with tf.variable_scope('value_loss'):
            self.value_loss = tf.reduce_mean(tf.square(tf.reshape(self.value, [-1]) - self.R))

        with tf.variable_scope('entropy_neg'):
            self.entropy_neg = tf.reduce_mean(tf.reduce_sum(self.policy * self.log_policy, axis=-1))

        with tf.variable_scope('ac_loss'):
            ent_coef = 0.01
            vf_coef=0.5
            self.ac_loss = self.policy_loss + self.entropy_neg * ent_coef + self.value_loss * vf_coef

        with tf.variable_scope('loss'):
            self.loss = self.ac_loss

        with tf.variable_scope('train'):
            self.LR = tf.placeholder(shape=[], dtype=self.cf.tf_dtype, name='LR')

            self.params = tf.trainable_variables('net_params') + tf.trainable_variables('fc_params') + \
                            tf.trainable_variables('logits_params') + tf.trainable_variables('value_params')
            grads = tf.gradients(self.loss, self.params)

            if self.cf.max_grad_norm is not None:
                grads, self.grad_norm = tf.clip_by_global_norm(grads, self.cf.max_grad_norm)
            grads = list(zip(grads, self.params))

            _trainer = tf.train.RMSPropOptimizer(learning_rate=self.LR, decay=0.99, epsilon=1e-5)
            self.trainer = _trainer.apply_gradients(grads)


class A2C(object):
    def __init__(self, config, sess):
        self.cf = config
        self.sess = sess

        self.eval_env = Env(self.cf)
        self.action_n = self.eval_env.action_n

        self.Pnet = ACnet(self.cf, action_n=self.action_n)


        self.saver = tf.train.Saver(max_to_keep=1)


    def predict_a_v(self, state):
        a, v = self.sess.run([self.Pnet.policy_action, self.Pnet.value], \
                            {self.Pnet.input: state})
        return a.flatten(), v.flatten()

    def train_net(self, step, states, actions, R, values):
        advantages = R - values

        lr = self.cf.learning_rate * (1. - 1.*step / self.cf.total_step)

        feed_dict = {self.Pnet.input: states, self.Pnet.action:actions, \
                    self.Pnet.LR: lr, \
                    self.Pnet.R: R, self.Pnet.advantages: advantages}

        _, ac_loss, grad_norm = self.sess.run([self.Pnet.trainer, self.Pnet.ac_loss, \
                                self.Pnet.grad_norm], feed_dict)
        # if grad_norm > self.cf.max_grad_norm:
        #     print(grad_norm)
        assert not np.isnan(ac_loss)
        self.ac_loss_avg.append(ac_loss)


    def learn(self):
        if not os.path.exists(self.cf.model_dir):
            os.makedirs(self.cf.model_dir)

        self.summary_write = tf.summary.FileWriter(self.cf.summary_dir, self.sess.graph)
        self.summary_write.flush()

        self.episode_r_summary = tf.Variable(0., trainable=False)
        era_op = tf.summary.scalar('reward/episode_r_avg', self.episode_r_summary)
        self.eval_episode_r_summary = tf.Variable(0., trainable=False)
        eera_op = tf.summary.scalar('reward/evaluate_episode_r_avg', self.eval_episode_r_summary)
        self.ac_loss_summary = tf.Variable(0., trainable=False)
        ala_op = tf.summary.scalar('loss/ac_loss_avg', self.ac_loss_summary)

        self.summary_op = tf.summary.merge([era_op, eera_op, ala_op])

        self.ac_loss_avg = []

        self.env = MultiEnv_thread(self.cf)

        print('\nLearning...\n')

        step = 0
        state = self.env.reset()
        action, value = self.predict_a_v(state)

        episodes_average = []
        episode_rewards = np.zeros(self.cf.nenvs)
        best_score = -9999.
        while step < self.cf.total_step:
            states_mb, actions_mb, rewards_mb, dones_mb, values_mb = [], [], [], [], []
            for i in range(self.cf.nsteps):
                step += self.cf.nenvs
                state_1, reward, done, real_done = self.env.act(action)
                states_mb.append(state)
                actions_mb.append(action)
                rewards_mb.append(np.sign(reward))
                dones_mb.append(done)
                values_mb.append(value)

                episode_rewards += reward
                episodes_average += episode_rewards[real_done].tolist()
                episode_rewards[real_done] = 0

                state = state_1
                action, value = self.predict_a_v(state)


            states_mb = np.asarray(states_mb).swapaxes(1, 0)
            states_mb = states_mb.reshape([-1] + list(states_mb.shape[2:]))
            actions_mb = np.asarray(actions_mb, dtype=np.int32).swapaxes(1, 0).flatten()
            rewards_mb = np.asarray(rewards_mb).swapaxes(1, 0)
            dones_mb = np.asarray(dones_mb, dtype=np.bool).swapaxes(1, 0)
            values_mb = np.asarray(values_mb).swapaxes(1, 0)

            for n, (r, d, v) in enumerate(zip(rewards_mb, dones_mb, value)):
                r = r.tolist() + [v]
                for i in range(len(d)-1, -1, -1):
                    r[i] += self.cf.discount * r[i+1] * (1 - d[i])
                rewards_mb[n] = r[:-1]

            rewards_mb = rewards_mb.flatten()
            values_mb = values_mb.flatten()

            self.train_net(step, states_mb, actions_mb, rewards_mb, values_mb)

            if step % self.cf.evaluate_every_step == 0:
                episode_r = np.array(episodes_average)
                print('step: ', step, 'episodes: ', episode_r.size, 'average: ', episode_r.mean(), 'std: ', episode_r.std())
                ac_l_a = np.array(self.ac_loss_avg)
                print('ac_loss: ', step,  'average: ', ac_l_a.mean(), 'std: ', ac_l_a.std())

                eval_episode_r = self.evaluate()
                summary_op = self.sess.run(self.summary_op, {self.episode_r_summary: episode_r.mean(), 
                                            self.eval_episode_r_summary: eval_episode_r.mean(), 
                                            self.ac_loss_summary:ac_l_a.mean()})
                self.summary_write.add_summary(summary_op, global_step=step)
                self.summary_write.flush()
                episodes_average = []
                self.ac_loss_avg = []

                with open(self.cf.summary_dir + 'r.csv', 'a') as f:
                    r_csv = str(time.time()) + ',' + str(step) + ',' + str(episode_r.mean()) + ',' + str(episode_r.std()) +\
                            ',' + str(eval_episode_r.mean()) + ',' + str(eval_episode_r.std()) +\
                            ',' + str(ac_l_a.mean()) + ',' + str(ac_l_a.std()) + '\n'
                    print(r_csv)
                    f.write(r_csv)

                if eval_episode_r.mean() > best_score:
                    best_score = eval_episode_r.mean()
                    self.saver.save(self.sess, self.cf.model_dir + str(step))
        self.env.close()


    def evaluate(self, load_model = False):

        if load_model:
            print('\nEvaluate...')
            ckpt_path = tf.train.get_checkpoint_state(self.cf.model_dir).model_checkpoint_path
            print('Loading Model', ckpt_path, '\n')
            self.saver.restore(self.sess, ckpt_path)

        env = self.eval_env
        env.reset()
        episode_step = 0
        episode_reward = 0
        episodes_average = []

        while len(episodes_average) < self.cf.evaluate_episodes:
            episode_step += 1
            action, _ = self.predict_a_v(env.recent_states)
            state, reward, done = env.act(action, is_training=False)
            episode_reward += reward

            if done or episode_step > self.cf.evaluate_episode_step:
                # print('evaluate episode_step: ', episode_step)
                env.real_done = True
                episodes_average.append(episode_reward)
                episode_step = 0
                episode_reward = 0
                env.reset()

        e_a = np.array(episodes_average)
        print('evaluate: ', 'episodes: ', e_a.size, 'average: ', e_a.mean(), 'std: ', e_a.std())
        return e_a


