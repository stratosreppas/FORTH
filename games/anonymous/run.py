from anonymous import Anonymous
import argparse
import numpy as np
from scipy.stats import norm
import sys
import os
import matplotlib.pyplot as plt
from instances.L4_anonymous import L4Anonymous

sys.setrecursionlimit(2000)


try:
    from global_utils import normalize, gaussian
except:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from global_utils import normalize, gaussian



parser = argparse.ArgumentParser(description='Run Anonymous Games!')
parser.add_argument('--name', default='anonymous', help='The name of the game')
parser.add_argument('--actions', default=2, help='The number of actions')
parser.add_argument('--players', default=600, help='The number of players')
parser.add_argument('--utility', default='gaussian', help='')
parser.add_argument("--cost", default=0.1, help="The cost of the game")
parser.add_argument('--epsilon', default=0.1, help='The epsilon for the approximate Nash equilibria')
parser.add_argument('--pure_nash', default=True, help='Find approximate pure Nash equilibria')
parser.add_argument('--plot', default=True, help='Plot the utility functions of the first couple of players')


if __name__ == '__main__':

    args = parser.parse_args()

    name = args.name
    actions = int(args.actions)
    players = int(args.players)
    utility = args.utility
    cost = float(args.cost)
    epsilon = float(args.epsilon)
    pure_nash = bool(args.pure_nash)
    plot = bool(args.plot)

    game = Anonymous(name, actions, players, utility, cost)
        # Plot the utility functions of the first couple of players
    
    if plot:
        for player in range(4):
            plt.plot(game.utility[player][0], label='Action 0')
            plt.plot(game.utility[player][1], label='Action 1')
            plt.legend()
            plt.show()

    if pure_nash:
        print(game.oblivious_ptas(epsilon))