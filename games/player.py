import numpy as np
from abc import abstractmethod

class Player():
    def __init__(self, name, actions, utility_params):
        self.name = name
        self.actions = actions
        self.n_actions = len(actions)
        self.utility_params = utility_params
    
    @abstractmethod
    def utility_function(self, *args, **kwargs):
        pass