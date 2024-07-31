from abc import abstractmethod
import logging
import sys 
import os
import gymnasium as gym
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from games.game import Game

class Bayesian(Game):
    
    def __init__(self, name, actions, players, utility):
        super().__init__(name, actions, players, utility)

        self.environment = None


