import sys
import os
import copy

from utils import permutations, MaxFlowGraph

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from game import Game

class AnonymousGame(Game):
    def __init__(self, name, actions, players, utility):
        super().__init__(name, actions, players, utility)

    def check_nash():
        pass

    def pure_nash(self, epsilon=0.1):
        '''
        Implementation of Daskalakis algorithm for finding pure Nash equilibria in Anonymous Games. 
        '''
        equilibria = []

        # Create the partitions of the players into sets of number of actions cardinality
        pi = permutations(self.players, self.actions)

        # Create the maximum flow graph
        g = MaxFlowGraph(self.players+self.actions+2)
        s = 0
        t = self.players + self.actions + 1
        for player in range(self.players):
            g.addEdge(s, player+1, 1)

        # For each partition, check if it is a Nash equilibrium
        for partition in pi:
            nash = copy.deepcopy(g)
            
            for i, action in enumerate(range(self.players+1, self.players+self.actions+1)):
                nash.addEdge(action, t, partition[i])
            # nash.plot_graph()

            for player in range(self.players):
                    max_utility = max(self.utility[player][action][partition[1]] for action in range(self.actions))
                    for action in range(self.actions):
                        if max_utility - self.utility[player][action][partition[1]] <= epsilon:
                            nash.addEdge(player, self.players + action + 1, 1)
            # nash.plot_graph()
            if nash.DinicMaxflow(s, t) == self.players:
                equilibria.append(partition)        

        return equilibria   
