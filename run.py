import gymnasium as gym
import numpy as np
import logging
import sys
import os
import argparse
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecMonitor, SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback


import games
from games.L23_markovian import L23Markovian


logging.basicConfig(filename='run.log', filemode='w', level=logging.INFO)
logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')

parser = argparse.ArgumentParser(description='Run Games!')
parser.add_argument('--game', default='l23_markovian', help='The game: ["l23_markovian", l4_anonymous"]')
parser.add_argument('--alg', default='ppo', help='-- For the case of The l23_markovian -- The algorithm: ["ppo", "recurrent_ppo", "a2c"]')
parser.add_argument('--training', default=True, help='-- For the case of The l23_markovian -- Train the agent')
parser.add_argument('--testing', default=False, help='-- For the case of The l23_markovian -- Test the agent')
parser.add_argument('--n_steps', default=100000, help='-- For the case of The l23_markovian -- Number of steps to train the agent')
parser.add_argument('--steps', default=10000, help='-- For the case of The l23_markovian -- Number of steps to save the model')

def get_agent(alg, game):
    ###This section creates games. We are using SubprocVecEnv for parallel computing to
    ### accelerate the learning process. Use only one environment in cluster mode. Use as many environment as your
    ### threads (or a little less) for fastest training.
    env = get_env(args.game)
    if game == 'l23_markovian':
        env.reset()
        _, _, _, _, info = env.step(0)
        info_keywords = tuple(info.keys())
        
        # envs = SubprocVecEnv(
        #     [
        #         lambda: get_env(args.game)
        #         for i in range(1)
        #     ]
        # )

        # envs = VecMonitor(envs, "vec_gym_results_", info_keywords=info_keywords)
        if(alg == 'ppo'):
            return PPO('MlpPolicy', env, verbose=1, tensorboard_log="./tensorboard/ppo_l23_markovian_tensorboard/")
        elif(alg == 'recurrent_ppo'):
            return PPO('MlpLstmPolicy', env, verbose=1, tensorboard_log="./tensorboard/rppo_l23_markovian_tensorboard")
        elif(alg == 'a2c'):
            return A2C('MlpPolicy', env, verbose=1, tensorboard_log="./tensorboard/a2c_l23_markovian_tensorboard/")


def get_env(use_case):
    print('Loading the environment...')
    if use_case == 'l23_markovian':
        return L23Markovian()

if __name__ == '__main__':
    args = parser.parse_args()
    game = str(args.game)
    alg = str(args.alg)
    training = bool(args.training)
    testing = bool(args.testing)
    n_steps = int(args.n_steps)
    steps = int(args.steps)

    name = game + "_" + alg + "_" + str(n_steps)

    # Callback
    checkpoint_callback = CheckpointCallback(save_freq=steps, save_path="logs/" + name, name_prefix=name)
    print('Loading the agent...')
    neuron = get_agent(alg, game)
    print(neuron)
    
    if training:
        print('Training the agent...')
        neuron.learn(total_timesteps=n_steps, callback=checkpoint_callback, tb_log_name=name + "_run")
        neuron.save(name)
    
    if testing:
        pass

