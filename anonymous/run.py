from anonymous import AnonymousGame
import argparse
import numpy as np
from scipy.stats import norm
import sys
import os
import matplotlib.pyplot as plt


try:
    from global_utils import normalize, gaussian
except:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from global_utils import normalize, gaussian



parser = argparse.ArgumentParser(description='Run Anonymous Games!')
parser.add_argument('--name', default='anonymous', help='The name of the game')
parser.add_argument('--actions', default=2, help='The number of actions')
parser.add_argument('--players', default=40, help='The number of players')
parser.add_argument('--utility', default='gaussian', help='')
parser.add_argument("--cost", default=0.1, help="The cost of the game")
parser.add_argument('--epsilon', default=0.1, help='The epsilon for the approximate Nash equilibria')
parser.add_argument('--pure_nash', default=True, help='Find approximate pure Nash equilibria')


def get_utility(utility, players, actions, cost):
    utility_function = np.zeros((players, actions, players+1))
    for player in range(players):
        if utility == 'gaussian':
            mean = np.random.randint(0, players)
            sigma = (2-1)*np.random.random_sample() + 1 # select sigma between 1 and 2
            utility_function[player][0] = np.array(gaussian(players+1, mean+1, sigma))
            utility_function[player][1] =np.array(gaussian(players+1, mean, sigma)) - cost
            utility_function[player] = normalize(utility_function[player])
        elif utility == 'uniform':
            return np.random.rand(players, actions)
        elif utility == 'random':
            return np.random.randint(0, 100, size=(players, actions))
        else:
            return np.zeros((players, actions))
    return utility_function

if __name__ == '__main__':

    args = parser.parse_args()

    name = args.name
    actions = int(args.actions)
    players = int(args.players)
    utility = args.utility
    cost = float(args.cost)
    epsilon = float(args.epsilon)
    pure_nash = bool(args.pure_nash)

    utility_function = get_utility(utility, players, actions, cost)
    # # Plot the utility functions of the first couple of players
    # for player in range(4):
    #     plt.plot(utility_function[player][0], label='Action 0')
    #     plt.plot(utility_function[player][1], label='Action 1')
    #     plt.legend()
    #     plt.show()

    game = AnonymousGame(name, actions, players, utility_function)
    if pure_nash:
        print(game.pure_nash(epsilon))