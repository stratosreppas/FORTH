import sys
import os
import numpy as np
import scipy

import matplotlib.pyplot as plt
from SALib.sample import saltelli
from SALib.analyze import sobol
import scipy.special

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from games.anonymous.anonymous import Anonymous
from games.player import Player
from global_utils import normalize, gaussian

NOT_FIRE = 0
FIRE = 1

class L4Neuron(Player):
    
        def __init__(self, name='L4Neuron', actions=[NOT_FIRE, FIRE], utility_params={'type': 'gaussian', 'cost': 0.1, 'mean_mean': 10, 'mean_std': 1}):
            super().__init__(name, actions, utility_params)

    
        def utility_function(self, n_players): 
            utility_function = np.zeros((self.n_actions, n_players))
            if self.utility_params['type'] == 'gaussian':
                mean = np.random.normal(self.utility_params['mean_mean'], self.utility_params['mean_std']) # select mean with deviation of 1
                # Based on the 68-95-99.7 rule, 99.7% of the values will be inside the scope we are examining and a step cannot contain more than 10% of the values of the gaussian
                std = np.random.uniform(np.sqrt(2)/scipy.special.erfinv(0.1), max(mean/3, (n_players-mean)/3))
                utility_function[0] = np.array(gaussian(n_players, mean, std))
                utility_function[1] =np.array(gaussian(n_players, mean-1, std)) - self.utility_params['cost']
                utility_function = normalize(utility_function)
            return utility_function

class L4Anonymous(Anonymous):

    def __init__(self, name='L4Anonymous', n_players=100, neuron_params=None):
        players=[L4Neuron(utility_params=neuron_params) if neuron_params is not None else L4Neuron() for _ in range(n_players)]

        self.utility = np.array([player.utility_function(n_players) for player in players])

        super().__init__(name=name, players=players)
        
        
    def sensitivity_analysis(self, median=None, std=None, cost=None):
        problem = {
            'num_vars': 3,
            'names': ['mean_mean', 'mean_std', 'cost'],
            'bounds': [[0, self.n_players],]
        }
        
        param_values = saltelli.sample(problem, 1000)
        Y = np.array([neuron.utility_function(params) for neuron in self.players for params in param_values])
        
        Si = sobol.analyze(problem, Y, print_to_console=True)
        
        # Plot first-order indices
        plt.bar(problem['names'], Si['S1'])
        plt.xlabel('Parameters')
        plt.ylabel('First-order Sobol Index')
        plt.title('Sensitivity Analysis')
        plt.show()



