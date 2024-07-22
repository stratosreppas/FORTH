from abc import abstractmethod
import logging
import sys 
import os
import gymnasium as gym
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from game import Game

class Markovian(gym.Env, Game):
    
    def __init__(self, name, actions, players, utility):
        super(Game, self).__init__(name, actions, players, utility)

        self.state = self.get_initial_state()
        self.observation_space = self.get_observation_space()
        self.action_space = gym.spaces.Discrete(self.actions)
        self.current_step = 0
        self.seed()

        
        logging.info("Markovian game created")
        logging.info("Name: " + self.name)
        logging.info("Actions: " + self.actions)
        logging.info("Players: " + self.players)
        logging.info("Utility: " + self.utility)
        logging.info("Initial state: " + self.state)
        logging.info("Observation space: " + self.observation_space)

        self.total_reward = 0
        self.episode_over = False
        self.time_start = 0

        self.execution_time = 0
        self.episode_count = 0
        self.file_results = "results.csv"
        self.obs_csv = self.name + "_observation.csv"

        self.info = {}


    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]
    

    @abstractmethod
    def get_observation_space(self):
        pass


    @abstractmethod
    def get_initial_state(self):
        pass


    @abstractmethod
    def step(self, action):
        pass


    def reset(self):
        """
        Resets the state of the environment and returns an initial observation.

        :return: The numpy array of an initial observation, using the get_state() function
        """
        self.current_step = 0
        self.total_reward = 0

        # episode over
        self.episode_over = False
        self.info = {}

        return np.array(self.get_state())
    

    def render(self, mode='human', close=False):
        """
        Render the environment to the screen. Not implemented.
        """
        return
    
    @property
    def get_reward(self):
        """
            Returns the reward for the current state.
        """
        return self.utility(self.state)

    

