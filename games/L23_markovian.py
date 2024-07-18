import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from game import Game
from markovian.markovian import Markovian

class L23Markovian(Markovian):

    def __init__(self, name, actions, players, utility):
        super(Markovian, self).__init__(name, actions, players, utility)

    def get_observation_space(self):
        pass

    def get_initial_state(self):
        pass

    def step(self, action):
        pass