
import os
import sys
os.chdir(sys.path[0])

import argparse
import tensorflow as tf

from config import Config
from agent import A2C


def make_hparam_string(cf):
    sub_dir = "%s,mgn_%s" % (cf.game, cf.max_grad_norm)
    if cf.debug: sub_dir = 'debug_' + sub_dir
    print('hyperparam: ' + sub_dir + '\n')
    return sub_dir


cf = Config()


def get_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--g', type=str, default=cf.game)
    
    parser.add_argument('--mgn', type=float, default=0)

    parser.add_argument('--d', type=int, default=int(cf.debug))
    parser.add_argument('--e', type=int, default=0) # evaluate
    parser.add_argument('--gpu', type=str, default='7')

    args = parser.parse_args()
    return args

args = get_args()
cf.game = args.g
cf.debug = bool(args.d)

if args.mgn != 0:
    cf.max_grad_norm = args.mgn


if bool(args.d):
    print('\nDebugging ...\n')

    cf.total_step = 4000
    cf.evaluate_every_step = 2000
    cf.evaluate_episodes = 1
    cf.nenvs = 1


subdir = make_hparam_string(cf)
cf.summary_dir += (subdir + '/')
cf.model_dir += (subdir + '/')

if not bool(args.e) and not bool(cf.debug):
    if os.path.exists(cf.summary_dir) or os.path.exists(cf.model_dir):
        print('parameter dirs exist\n')
        sys.exit()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def main():
    sess = tf.Session(config=cf.tf_config)

    agent = A2C(cf, sess)
    sess.run(tf.global_variables_initializer())

    if bool(args.e):
        agent.evaluate(load_model=True)
    else:
        agent.learn()
    
    sess.close()


if __name__ == '__main__':
    main()




