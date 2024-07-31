import sys
import os
import copy
import numpy as np

try:
    from utils import compositions, MaxFlowGraph
except:
    from anonymous.utils import compositions, MaxFlowGraph

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from games.game import Game
from global_utils import normalize, gaussian

class AnonymousGame(Game):
    '''
    Below is an implementation of Daskalakis PTAS algorithms for anonymous games.
    The algorithm is described in the following paper:
    "Approximate Nash Equilibria in Anonymous Games" by Constantinos Daskalakis.
    https://people.csail.mit.edu/costis/anonymousJET.pdf
    '''
    def __init__(self, name='AnonymousGame', actions=2, players=100, utility='gaussian', cost=0.1):
        super().__init__(name, actions, players, self.get_utility(utility, players, actions, cost))

    def pure_nash(self, epsilon=0.1):
        '''
        Implementation of Daskalakis algorithm for finding pure Nash equilibria in Anonymous Games. 
        '''
        equilibria = []

        # Create the partitions of the players into sets of number of actions cardinality
        partitions = compositions(self.players, self.actions)
        sigma = [0, 1]

        for theta in partitions:
            if self.find_nash(theta, sigma, epsilon):
                equilibria.append(theta)  

        return equilibria

    def oblivious_ptas(self, epsilon=0.1):
        equilibria = []

        # Step 1
        C=0.8
        k = int(np.floor(C/epsilon))
        print(k)

        # Step 2
        for t in range(1, self.players+1): # This is the number of players that play mixed strategies 
            for t0 in range(self.players-t): # This is the number of players that only play the first action (0) - not fire
                t1 = self.players - t - t0 # This is the number of players that only play the second action (1) - fire
                t = self.players
                t0 = 0
                t1 = 0
                # Step 3a
                if(t > k**3):
                    print('a')
                    print('t0:', t0, 't1:', t1, 't:', t)
                    for i in range(1, k*self.players+1):
                        # So far, we have a complexity of O(kn^3)
                        # Choose an integer multiple (j) of 1/(kn)
                        q = i/(k*self.players)

                        theta = [t0, t1, t]
                        sigma = [0, 1, q]
                        
                        # Calculate the expected utilities of the players for the various strategies
                        # In O(n^3) time
                        print('Computing expected utilities')
                        exp_utility = self.compute_exp_utilities(theta, sigma)
                        print('Done computing expected utilities!')

                        # Create the max flow graph
                        g = MaxFlowGraph(self.players+len(sigma)+2) # {[n] U σ = {0,1,q} U {s,t}}

                        source = 0
                        target = self.players + len(sigma) + 1

                        # Creation of the N edges from s to [n] with capacity 1
                        for player in range(1,self.players+1):
                            g.addEdge(source, player, 1)
                        
                        # Creation of the O(N) edges from [n] to σ = {0,1,q} with capacity 1
                        for player in range(1, self.players+1):
                            max_utility = max(exp_utility[player-1][strategy] for strategy in sigma)
                            for i, strategy in enumerate(sigma):
                                if(theta[i] > 0 and max_utility - exp_utility[player-1][strategy] <= epsilon):
                                    g.addEdge(player, self.players + 1 + i, 1)
                        
                        # Creation of the |σ| edges from σ = {0,1,q} to t with capacity θ_σ
                        for strategy, partition in enumerate(theta):
                            g.addEdge(self.players + 1 + strategy, target, partition)

                        if g.DinicMaxflow(source, target) == self.players:
                            equilibria.append(sigma)
                            print(sigma)
                            # break                
                # Step 3b
                else:
                        print('b')
                        print('t0:', t0, 't1:', t1, 't:', t)
                        # Choose an integer multiple (j) of 1/(kn)
                        q = [i/(k**2) for i in range(1, k**2)]

                        sigma = [0,1] + q
                        
                        print('Calculating partitions')
                        partitions = compositions(t, k**2-1)
                        print('Done calculating partitions!')
                        
                        for y in partitions:
                            theta = [t0, t1] + y
                            # Calculate the expected utilities of the players for the various strategies
                            print('Computing expected utilities')
                            exp_utility = self.compute_exp_utilities(theta, sigma)
                            print('Done computing expected utilities!')                            
                            # Create the max flow graph
                            g = MaxFlowGraph(self.players+len(sigma)+2) # {[n] U σ = {0,1,q} U {s,t}}

                            source = 0
                            target = self.players + len(sigma) + 1

                            # Creation of the N edges from s to [n] with capacity 1
                            for player in range(1,self.players+1):
                                g.addEdge(source, player, 1)
                            
                            # Creation of the O(N) edges from [n] to σ = {0,1,q} with capacity 1
                            for player in range(1, self.players+1):
                                max_utility = max(exp_utility[player-1][strategy] for strategy in sigma)
                                for i, strategy in enumerate(sigma):
                                    if(theta[i] > 0 and max_utility - exp_utility[player-1][strategy] <= epsilon):
                                        g.addEdge(player, self.players + 1 + i, 1)
                            
                            # Creation of the |σ| edges from σ = {0,1,q} to t with capacity θ_σ
                            for strategy, partition in enumerate(theta):
                                g.addEdge(self.players + 1 + strategy, target, partition)

                            if g.DinicMaxflow(source, target) == self.players:
                                equilibria.append(sigma)
                                print(sigma)
                                break


        return equilibria
            

    def compute_exp_utilities(self, theta, sigma):
        '''
        The dp algorithm proposed in Daskalakis paper for computing expected utilities in Anonymous Games. 

        We will add a small modification: because depending on which strategy the player that we examine plays, 
        the probabilities of players playing a strategy will be different, since we have to take into account the
        fact that the player that we examine plays a strategy, effectively removing a strategy from the pool of 
        strategies that the other players can play. So we will add another dimension to the dp table, which will
        signify the strategy that the player plays.

        It is implemented for the specific case of 2 actions ([ξ] = 2) and has a complexity of O(n^2).

        Parameters:
        theta (tuple): The tuple containing the number of players that play the 3 types of strategies - not fire, fire, mixed
        q (float): The probability of the mixed strategy

        Returns:
        exp_utilities (list): The expected utilities of the players for the various strategies
        '''
        T = [[0 for _ in range(self.players)] for _ in range(self.players-1)]
        delta = [] # It signifies the probability of a player to play the second action (1) - fire
        # Initialization of delta
        for i, players in enumerate(theta):
            for _ in range(players):
                delta.append(sigma[i])
        
        probs = {}
        
        # We will calculate the dp table for each strategy that the player that we examine could play
        for strategy in sigma:
            # We will remove the strategy of the player that we examine
            if strategy not in delta:
                probs[strategy] = [0 for _ in range(self.players)]
                continue

            delta.remove(strategy)

            # Initialization of dp table T
            T[0][1] = delta[0]
            T[0][0] = (1-delta[0])

            # Recursive dp formula
            # The final row of the dp table contains the probabilities of the players to play the second action (1) - fire
            # if the player that we examine plays the strategy that we are considering
            for i in range(1, self.players-1):
                for l in range(self.players):
                    if (0 < l < i):
                        T[i][l] = delta[i]*T[i-1][l-1] + (1-delta[i])*T[i-1][l]

                    elif (l == i):
                        T[i][l] = delta[i]*T[i-1][l-1]

                    elif (l == 0): 
                        T[i][l] = (1-delta[i])*T[i-1][l]

                    elif (l > i):
                        T[i][l] = 0
        
            # Add the removed strategy back to delta to be considered in the next iteration
            delta.append(strategy)

            # We store it in the probs list 
            probs[strategy] = T[self.players-2]
            # print(probs[strategy], strategy)
             
        # Calculate the expected utilities of the players for the various strategies    
        exp_utilities = [{}  for _ in range(self.players)]
        for i in range(self.players):
            for strategy in sigma:
                exp_utilities[i][strategy] = strategy * np.dot(self.utility[i][1], np.array(probs[strategy])) + (1-strategy) * np.dot(self.utility[i][0], np.array(probs[strategy]))
                # if(i == 1 and (strategy == 0 or strategy == 1)): 
                #     print(exp_utilities[i][strategy], strategy)
        return exp_utilities
    
    def lets_see(self, epsilon=0.1, n_probs=5):
        
        sigma=[i/n_probs for i in range(1, n_probs+1)]
        print('Computing compositions')
        thetas = compositions(self.players, n_probs)
        print(len(thetas))
        equilibria = []
        for theta in thetas:
            if self.find_nash(theta, sigma, epsilon):
                print(theta)
                equilibria.append(theta)
        
        return equilibria
            
            
    def find_nash(self, theta, sigma, epsilon=0.1):
        '''
        Function that finds a Nash equilibrium in an Anonymous Game using the max flow algorithm.

        Parameters:
        theta (tuple): The tuple containing the number of players that play the different types of strategies
        sigma (list): The list containing the strategies that the players play - aka the probabilities 
                    of the players playing the second action (1) - fire
        epsilon (float): The approximation parameter for the algorithm

        Returns:
        bool: True if a Nash equilibrium is found, False otherwise
        '''
        # print('Computing expected utilities')
        exp_utility = self.compute_exp_utilities(theta, sigma)
        # print('Done computing expected utilities!')

        # print('Solving the max flow problem') 
        g = MaxFlowGraph(self.players+len(sigma)+2) # {[n] U σ U {s,t}}

        source = 0
        target = self.players + len(sigma) + 1

        # Creation of the N edges from s to [n] with capacity 1
        for player in range(1,self.players+1):
            g.addEdge(source, player, 1)
        
        # Creation of the O(N) edges from [n] to σ with capacity 1
        for player in range(1, self.players+1):
            max_utility = max(exp_utility[player-1][strategy] for strategy in sigma)
            for i, strategy in enumerate(sigma):
                if(theta[i] > 0 and max_utility - exp_utility[player-1][strategy] <= epsilon):
                    g.addEdge(player, self.players + 1 + i, 1)
        
        # Creation of the |σ| edges from σ to t with capacity θ_σ
        for strategy, partition in enumerate(theta):
            g.addEdge(self.players + 1 + strategy, target, partition)

        return(g.DinicMaxflow(source, target) == self.players)
    
    def get_utility(self, utility, players, actions, cost):
        utility_function = np.zeros((players, actions, players))
        for player in range(players):
            if utility == 'gaussian':
                mean = np.random.randint(players)
                sigma = (20-10)*np.random.random_sample() + 10 # select sigma between 1 and 2
                utility_function[player][0] = np.array(gaussian(players, mean, sigma))
                utility_function[player][1] =np.array(gaussian(players, mean+1, sigma)) - cost
                utility_function[player] = normalize(utility_function[player])
            elif utility == 'uniform':
                return np.random.rand(players, actions)
            elif utility == 'random':
                return np.random.randint(0, 100, size=(players, actions))
            else:
                return np.zeros((players, actions))
        return utility_function
