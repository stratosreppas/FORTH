from abc import abstractmethod
import logging
import sys 
import os
import gymnasium as gym
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from games.game import Game

class Markovian(gym.Env, Game):
    
    def __init__(self, name, players):
        super().__init__(name, players)

        self.state = self.get_initial_state()
        self.observation_space = self.get_observation_space() # The multi-agent observation space will be set in the get_observation_space() function
        self.action_space = gym.spaces.Discrete(self.n_actions) # We assume that the agents have the same actions (2 in our case)
        self.current_step = 0
        self.seed()

        logging.info("Markovian game created!")
        logging.info("Name: " + str(self.name))
        logging.info("Players: " + str(self.n_players))
        logging.info("Initial state: " + str(self.state))
        logging.info("Observation space: " + str(self.observation_space))

        self.total_reward = 0
        self.done = []
        self.time_start = 0

        self.execution_time = 0
        self.episode_count = 0
        self.file_results = "results.csv"
        self.obs_csv = self.name + "_observation.csv"

        self.info = {}

        for i, player in enumerate(self.players):
            self.done.append(False)
            self.info[i] = {}
            self.info[i]["total_reward"] = 0
            self.info[i]["total_reward_episode"] = 0
            self.info[i]["total_reward_episode_list"] = []
            self.info[i]["total_reward_list"] = []
            self.info[i]["total_reward_episode_list"].append(0)
            self.info[i]["total_reward_list"].append(0)
            self.info[i]['actions'] = self.actions


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
    def step(self, action, player):
        pass


    def reset(self, seed=None):
        """
        Resets the state of the environment and returns an initial observation.

        :return: The numpy array of an initial observation, using the get_state() function
        """
        # Custom reset for the possible variations
        self.custom_reset()

        self.current_step = 0
        self.total_reward = 0

        # episode over
        self.done = [False for _ in range(self.n_players)]
        self.info = {}

        return self.normalized_state(self.get_state()), self.info
    
    def render(self, mode='human', close=False):
        """
        Render the environment to the screen. Not implemented.
        """
        return
    
    @abstractmethod
    def get_reward(self):
        """
            Returns the reward for the current state.
        """
        pass

    
    @abstractmethod
    def get_state(self):
        """
            Returns the current state of the environment.
        """
        pass


    @abstractmethod
    def custom_reset(self):
        """
        Custom reset function for the game. 
        """
        pass


    def normalized_state(self, state):
        return (state-self.observation_space.low)/(self.observation_space.high - self.observation_space.low)


    

