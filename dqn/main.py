
import os
import sys
os.chdir(sys.path[0])

import argparse
import tensorflow as tf

from config import Config
from memory import Memory
from env import Env
from agent import DQN


def make_hparam_string(cf):
    sub_dir = "%s,mgn_%s" % (cf.game, cf.max_grad_norm)

    print('hyperparam: ' + sub_dir + '\n')
    return sub_dir


cf = Config()


def get_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--g', type=str, default=cf.game) #
    parser.add_argument('--lr', type=float, default=cf.learning_rate)
    parser.add_argument('--mgn', type=float, default=cf.max_grad_norm)
    parser.add_argument('--uf', type=float, default=cf.update_frequency)
    parser.add_argument('--fe', type=float, default=cf.final_explore)
    parser.add_argument('--fes', type=float, default=cf.final_explore_step)
    parser.add_argument('--ms', type=float, default=cf.memory_size) #
    parser.add_argument('--mss', type=float, default=cf.memory_start_size)
    parser.add_argument('--h', type=int, default=int(cf.huber_loss))
    parser.add_argument('--ee', type=float, default=cf.evaluate_explore)

    parser.add_argument('--d', type=int, default=int(cf.debug))
    parser.add_argument('--e', type=int, default=0) # evaluate
    parser.add_argument('--gpu', type=str, default='7')

    args = parser.parse_args()
    return args

args = get_args()
cf.game = args.g
cf.learning_rate = args.lr
cf.max_grad_norm = args.mgn
cf.update_frequency = int(args.uf)
cf.final_explore = args.fe
cf.final_explore_step = int(args.fes)
cf.memory_size = int(args.ms)
cf.memory_start_size = int(args.mss)
cf.huber_loss = bool(args.h)
cf.evaluate_explore = args.ee


if bool(args.d):
    print('\nDebugging ...\n')
    # cf.debug = bool(args.d)
    cf.final_explore_step = 1000
    cf.memory_size = 1500
    cf.memory_start_size = 500

    cf.total_step = 3000
    cf.evaluate_every_step = 1500
    cf.evaluate_episodes = 1


subdir = make_hparam_string(cf)
cf.summary_dir += subdir + '/'
cf.model_dir += subdir + '/'

if not bool(args.e) and not bool(args.d):
    if os.path.exists(cf.summary_dir) or os.path.exists(cf.model_dir):
        raise Exception('parameter dirs exist')

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def main():
    sess = tf.Session(config=cf.tf_config)

    dqn = DQN(cf, sess)
    sess.run(tf.global_variables_initializer())

    if bool(args.e):
        dqn.evaluate(load_model=True)
    else:
        dqn.learn()
    
    sess.close()



if __name__ == '__main__':
    main()




