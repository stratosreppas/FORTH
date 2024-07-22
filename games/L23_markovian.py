import sys
import os
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from game import Game
from markovian.markovian import Markovian
from global_utils import indicator_function

class L23Markovian(Markovian):

    def __init__(self, name, actions, players, utility):
        super(Markovian, self).__init__(name, actions, players, utility)

    def get_observation_space(self):
        return np.random.randint(0, 15, size=(1, 1))

    def get_initial_state(self):
        pass

    def step(self, action):
        """
        Step function for the game. This function should return the next state, reward, and whether the game is over.
        """
        return self.state, self.get_reward, self.episode_over
    
    def utility(self, state, action):
        """
        Utility function for the game. This function should return the utility of the state and action.
        """
        return a[i]*indicator_function(abs(action - indicator_function(state!=angle[i]))>0)
    
