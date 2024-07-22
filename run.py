import gymnasium as gym
import numpy as np
import logging
import sys
import os
import argparse
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecMonitor, SubprocVecEnv

import games
from games.L23_markovian import L23Markovian


logging.basicConfig(filename='run.log', filemode='w', level=logging.INFO)
logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')

parser = argparse.ArgumentParser(description='Run Games!')
parser.add_argument('--game', default='l23_markovian', help='The game: ["l23_markovian", l4_anonymous"]')
parser.add_argument('--alg', default=None, help='-- For the case of The l23_markovian -- The algorithm: ["ppo", "recurrent_ppo", "a2c"]')

def get_agent(alg, game):
    if game == 'l23_markovian':
        game.reset()
        _, _, _, _, info = game.step([0, 0])
        info_keywords = tuple(info.keys())
        
        envs = SubprocVecEnv(
            [
                lambda: get_env(args.use_case)
                for i in range(8)
            ]
        )

        envs = VecMonitor(envs, "vec_gym_results_", info_keywords=info_keywords)
        if(alg == 'ppo'):
            return PPO('MlpPolicy', envs, verbose=1, tensorboard_log="./ppo_l23_markovian_tensorboard/")
        elif(alg == 'recurrent_ppo'):
            return PPO('MlpLstmPolicy', envs, verbose=1, tensorboard_log="./ppo_l23_markovian_tensorboard")
        elif(alg == 'a2c'):
            return A2C('MlpPolicy', envs, verbose=1, tensorboard_log="./ppo_l23_markovian_tensorboard/")
        ###This section creates games. We are using SubprocVecEnv for parallel computing to
        ### accelerate the learning process. Use only one environment in cluster mode. Use as many environment as your
        ### threads (or a little less) for fastest training.

def get_env(use_case):
    if use_case == 'l23_markovian':
        return L23Markovian()

if __name__ == '__main__':
    args = parser.parse_args()
    game = args.game
    alg = args.alg
