from abc import abstractmethod
import logging
import sys 
import os
import gymnasium as gym
import numpy as np
import pettingzoo as pz
from pettingzoo import AECEnv
from pettingzoo.utils import wrappers, agent_selector

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from games.game import Game

class MARL(AECEnv, Game):

    metadata = {
        'render.modes': ['human'],
        'name': 'Markovian'
        }

    def __init__(self, name, players, seed=42, episode_steps=100):
        
        Game.__init__(self, name, players)
        AECEnv.__init__(self)

        self.timestep = 0
        self.num_moves = 0

        self.agents = [player.name for player in players]
        self.possible_agents = [player.name for player in players]
        self.agent_selector = agent_selector(agent_order=self.agents)
        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents))))
        )

        self.observation_spaces = {agent: self.observation_space(agent) for agent in self.agents}
        self.action_spaces = {agent: self.action_space(agent) for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.rewards = {}
        self.max_cycles = episode_steps
        
        self.seed = seed

        self.infos = {
            agent: {
            "total_reward": 0,
            "episode_reward": 0,
            "total_reward_episode_list": [0],
            "total_reward_list": [0],
            "actions": self.action_space(agent)
            } for agent in self.agents
        }

        logging.info("Markovian game created!")
        logging.info("Name: " + str(self.name))
        logging.info("Players: " + str(self.n_players))
        logging.info("Initial state: " + str(self.state))
        logging.info("Observation space: " + str(self.observation_spaces))
        logging.info("Action space: " + str(self.action_spaces))
        logging.info("Seed: " + str(self.seed))

        self.time_start = 0
        self.execution_time = 0
        self.file_results = "results.csv"
        self.obs_csv = str(self.name) + "_observation.csv"

    def _seed(self, seed=None):
        """
        Set the seed for the random number generator.
        Args:
            seed (int): The seed value to set for the random number generator. If None, a random seed will be used.
        Returns:
            list: A list containing the seed value that was set.
        """
        
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]
    
    @abstractmethod
    def get_observation_space(self):
        """
        Returns the observation space for the multi-agent reinforcement learning environment.
        """
        pass
    
    @abstractmethod
    def get_action_space(self):
        """
        Returns the action space of the environment.
        """
        pass

    @abstractmethod
    def observe(self, agent):
        """
        Observes the environment from the perspective of the agent.
        
        Parameters:
            agent (object): The agent that observes.
        Returns:
            The observation of the environment for the agent (determined by the subclass).
        """
        pass

    @abstractmethod
    def step(self, action):
        """
        Takes in an action for the current agent (specified by
        agent_selection) and needs to update
        - rewards
        - _cumulative_rewards (accumulating the rewards)
        - terminations
        - truncations
        - infos
        - agent_selection (to the next agent)
        And any internal state used by observe() or render()
        """
        pass

    def reset(self, seed=None, options=None):
        """
        Resets the state of the environment and returns an initial observation.

        :return: The numpy array of an initial observation, using the get_state() function
        """
        # Custom reset for the possible variations
        self.custom_reset()

        self.agents = self.possible_agents[:]
        self.agent_selection = self.agent_selector.reset()

        self.num_moves = 0
        self.total_reward = 0
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.state = {agent: None for agent in self.agents}
        self.observations = {agent: None for agent in self.agents}
        self.rewards = {agent: 0 for agent in self.agents} 
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
       

        # episode over
        self.done = False
        self.infos = {a: {} for a in self.agents}
    
    def normalized_obs(self, obs, agent):
        # =return obs
        return (obs-self.observation_space(agent).low)/(self.observation_space(agent).high - self.observation_space(agent).low)


    def render(self, mode='human', close=False):
        """
        Render the environment to the screen. Not implemented.
        """
        return
    
    @abstractmethod
    def get_reward(self, agent):
        pass

    def get_state(self):
        """
            Returns the current state of the environment.
        """
        return self.state

    def custom_reset(self):
        """
        Custom reset function for the game. 
        """
        pass
    
    @abstractmethod
    def save_obs_to_csv(self):
        pass


    

