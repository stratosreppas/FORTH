import gymnasium as gym
import numpy as np
import logging
import sys
import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt

import ray.train
import supersuit as ss
from pettingzoo.butterfly import pistonball_v6
from stable_baselines3 import PPO, A2C

import ray
from ray import tune
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv
from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.logger import UnifiedLogger
from ray.tune.tuner import Tuner


import games
from instances.L23_markovian import L23Markovian
from instances.L4_anonymous import L4Anonymous
from instances.marl import MARL

log_file = 'run.log'
logging.basicConfig(filename=log_file, filemode='w', level=logging.INFO)
logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')

parser = argparse.ArgumentParser(description='Run Games!')
parser.add_argument('--game', default='marl', help='The game: ["l23_markovian", l4_anonymous, marl"]')
parser.add_argument('--alg', default='ppo', help='-- For the case of The l23_markovian/marl -- The algorithm: ["ppo", "recurrent_ppo", "a2c"]')
parser.add_argument('--training', default=False, help='-- For the case of The l23_markovian/marl -- Train the agent')
parser.add_argument('--testing', default=False, help='-- For the case of The l23_markovian/marl -- Test the agent')
parser.add_argument('--n_steps', default=100000, help='-- For the case of The l23_markovian/marl -- Number of steps to train the agent')
parser.add_argument('--steps', default=10000, help='-- For the case of The l23_markovian/marl -- Number of steps to save the model')
parser.add_argument('--episode_steps', default=100, help='-- For the case of The l23_markovian/marl -- Number of steps per episode')   
parser.add_argument('--tensorboard_log', default='./tensorboard', help='-- For the case of RL games -- Tensorboard log path')
parser.add_argument('--test_path', default='l23_markovian_ppo_100000.zip', help='-- For the case of RL games -- Path to the model to test')
parser.add_argument('--plot', default=True, help='-- Plot the results')
parser.add_argument('--pure', default=True, help='-- For the case of The l4_anonymous -- Find approximate pure Nash equilibria')
parser.add_argument('--mixed', default=True, help='-- For the case of The l4_anonymous -- Find approximate mixed Nash equilibria')
parser.add_argument('--epsilon', default=0.1, help='-- For the case of The l4_anonymous -- The epsilon for the approximate Nash equilibria')
parser.add_argument('--seed', default=42, help='-- Seed for the environment')   

def get_agent(alg, game):
    ###This section creates games. We are using SubprocVecEnv for parallel computing to
    ### accelerate the learning process. Use only one environment in cluster mode. Use as many environment as your
    ### threads (or a little less) for fastest training.
    env = get_instance(args.game)
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
            return PPO('MlpPolicy', env, verbose=1, tensorboard_log="./tensorboard/ppo_l23_markovian_tensorboard")
        elif(alg == 'recurrent_ppo'):
            return PPO('MlpLstmPolicy', env, verbose=1, tensorboard_log="./tensorboard/rppo_l23_markovian_tensorboard")
        elif(alg == 'a2c'):
            return A2C('MlpPolicy', env, verbose=1, tensorboard_log="./tensorboard/a2c_l23_markovian_tensorboard/")

        if(game == 'l4_anonymous'):
            return(L4Anonymous())
            
def test_model(model, env, n_episodes, n_steps, smoothing_window, fig_name):
    episode_rewards = []
    reward_sum = 0
    obs, _ = env.get_initial_state()
    print(obs)

    print("------------Testing ( smoothing window", smoothing_window, "episodes", n_episodes, ") -----------------")

    for e in range(n_episodes):
        for _ in range(n_steps):
            action, _ = model.predict(obs)
            obs, reward, done, _ = env.step(action)
            reward_sum += reward
            if done:
                episode_rewards.append(reward_sum)
                print("Episode {} | Total reward: {} |".format(e, str(reward_sum)))
                reward_sum = 0
                obs, _ = env.get_initial_state()
                break

    # Free memory
    env.close()
    del model, env

    # Plot the episode reward over time
    plt.figure()
    rewards_smoothed = pd.Series(episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()
    plt.plot(rewards_smoothed)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.savefig(fig_name, dpi=250, bbox_inches='tight')

    
def get_load_model(alg, tensorboard_log, test_path):
    if alg == 'ppo':
        return PPO.load(test_path, tensorboard_log=tensorboard_log)
    elif alg == 'rppo':
        return PPO.load(test_path, tensorboard_log=tensorboard_log)
    elif alg == 'a2c':
        return A2C.load(test_path, tensorboard_log=tensorboard_log)
    else:
        logging.error('Invalid algorithm!')

def get_instance(game='marl'):
    if game == 'l23_markovian':
        return L23Markovian()
    elif game == 'l4_anonymous':
        return L4Anonymous()
    elif game == 'marl':
        env = pistonball_v6.env(
            n_pistons=20,
            time_penalty=-0.1,
            continuous=True,
            random_drop=True,
            random_rotate=True,
            ball_mass=0.75,
            ball_friction=0.3,
            ball_elasticity=1.5,
            max_cycles=125,
            render_mode="rgb_array",
        )
        env = ss.color_reduction_v0(env, mode="B")
        env = ss.dtype_v0(env, "float32")
        env = ss.resize_v1(env, x_size=84, y_size=84)
        env = ss.normalize_obs_v0(env, env_min=0, env_max=1)
        env = ss.frame_stack_v1(env, 3)
        env.reset()
        return env
        return MARL()
    else:
        logging.error('Invalid game!')

if __name__ == '__main__':
    
    args = parser.parse_args()
    game = str(args.game)
    alg = str(args.alg)
    training = bool(args.training)
    testing = bool(args.testing)
    n_steps = int(args.n_steps)
    steps = int(args.steps)
    episode_steps= int(args.episode_steps)
    tensorboard_log = str(args.tensorboard_log)
    test_path = str(args.test_path)
    plot = bool(args.plot)
    pure = bool(args.pure)
    mixed = bool(args.mixed)
    epsilon = float(args.epsilon)
    seed = int(args.seed)

    name = game + "_" + alg + "_" + str(n_steps)
   
    instance = get_instance(game)

    if(game == 'l4_anonymous'):     

        if plot:
            for player in range(4):
                plt.plot(instance.utility[player][0], label='Action 0')
                plt.plot(instance.utility[player][1], label='Action 1')
                plt.legend()
                plt.show()  
        
        if(pure):
            # Redirect prints to log file
            with open(log_file, 'a') as f:
                sys.stdout = f
                print('Pure equilibria: {}'.format(instance.pure_nash(epsilon)))
            sys.stdout = sys.__stdout__  # Restore stdout

        if(mixed):
            print()
            # Redirect prints to log 
            equilibria = instance.mixed_nash_parallelised(epsilon, 3)
            with open(log_file, 'a') as f:
                sys.stdout = f
                print('Mixed equilibria: {}'.format(equilibria))
            sys.stdout = sys.__stdout__  # Restore stdout
            
            
            # Create 3D plot
            if(plot):
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')

                # Extract x, y, z coordinates from equilibria list
                x = [eq[0] for eq in equilibria]
                y = [eq[1] for eq in equilibria]
                z = [eq[2] for eq in equilibria]

                # Plot the points
                ax.scatter(x, y, z)

                # Set labels for x, y, z axes
                ax.set_xlabel('0.25')
                ax.set_ylabel('0.5')
                ax.set_zlabel('0.75')

                # Show the plot
                plt.show()
    elif(game == 'marl'):
        ray.init()
        
        env = get_instance(game)
           
        register_env("marl", lambda config: PettingZooEnv(get_instance(game)))
        
        policies = { 
            agent: (
            None,
            env.observation_space(agent),
            env.action_space(agent),
            {
                "model": {
                    "fcnet_hiddens": [256, 256],
                    "fcnet_activation": "relu",
                    "use_lstm": True,
                }
            }
            ) for agent in env.agents
        }
                
        def policy_mapping_fn(agent_id, episode, worker, **kwargs):
            # print(f"Additional arguments: {agent_id}")
            # print(f"Additional arguments: {episode}")
            # print(f"Additional arguments: {worker}")
            return agent_id  # Map each agent to its own policy
    
        
        config = (
            PPOConfig()
            .environment("marl")
            .env_runners(num_env_runners=1)
            .framework("torch")
            .multi_agent(
            policies=policies,
            policy_mapping_fn=policy_mapping_fn,
            )
        )

        print(config)
            
        tuner = Tuner(
            "PPO",
            param_space=config,  # Convert config to dictionary
            run_config=ray.train.RunConfig(
            stop={"timesteps_total": n_steps},
            log_to_file=True,
            storage_path=f'file://{os.path.abspath("./tensorboard")}',
            ),
        )

        print("Training... Log into tensorboard or view")
        tuner.fit()

   
        ray.shutdown()
        
