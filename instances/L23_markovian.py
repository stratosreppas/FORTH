import sys
import os
import numpy as np
from gymnasium import spaces

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from games.game import Game
from markovian.markovian import Markovian
from anonymous.anonymous import AnonymousGame
from global_utils import indicator_function, normalize


# STATES
ANGLE_0 = 0
ANGLE_22_5 = 1
ANGLE_45 = 2
ANGLE_67_5 = 3
ANGLE_90 = 4
ANGLE_112_5 = 5
ANGLE_135 = 6
ANGLE_157_5 = 7
ANGLE_180 = 8
ANGLE_202_5 = 9
ANGLE_225 = 10
ANGLE_247_5 = 11
ANGLE_270 = 12
ANGLE_292_5 = 13
ANGLE_315 = 14
ANGLE_337_5 = 15

# ACTIONS
NOT_FIRE = 0
FIRE = 1

MAX_STEPS = 25


class Player():
    def __init__(self, name, actions):
        self.name = name
        self.actions = actions
        self.angle = np.random.randint(ANGLE_0, ANGLE_225)
        self.a = np.random.uniform(0, 1)


class L23Markovian(Markovian):

    def __init__(self, name='L23Markovian', actions=2, L23players=1, L4players=100, utility=None, cost=0.1):
        self.anonymous = AnonymousGame(players=L4players)
        self.L4_neurons = L4players
        self.neurons = L23players
        self.neuron = Player('neuron', actions)
        print('Neuron', self.neuron.angle)
        self.equilibrium = []
        self.environment = {}
        self.cost = cost
        
        super().__init__(name, actions, L23players, utility)
 

    def get_observation_space(self):
        return spaces.Box(
            low=np.tile(np.array([0]), self.neurons), 
            high=np.tile(np.array([self.L4_neurons]), self.neurons), 
            dtype=np.float32
        )

    def get_state(self):
        angle = np.random.randint(ANGLE_0, ANGLE_225)
        self.environment['angle'] = angle
        coffirings = np.random.binomial(n=self.L4_neurons, p=1/16) # L4 output if they fire completely at random
        # print('Coffirings', coffirings)
        return self.equilibrium[FIRE] if self.neuron.angle==angle else coffirings


    def get_initial_state(self):
        return self.reset()
    
    def normalized_state(self, state):
        return (state-self.observation_space.low)/(self.observation_space.high - self.observation_space.low)

    def step(self, action):
        """
        Step function for the game. This function should return the next state, reward, and whether the game is over.
        """
        self.current_step += 1
        self.done = self.current_step >= MAX_STEPS
        reward = self.get_reward(action)
        self.state = self.normalized_state(self.get_state())
        self.total_reward += reward
        return self.state, reward, self.done, False, self.info
    

    def utility_function(self, state, action, player, environment):
        """
        Utility function for the game. This function should return the utility of the state and action.
        Exists only to state the game's involvement and its dependency from the values 
        """
        # print('Reward', player.a*indicator_function(abs(action - indicator_function(environment['angle']!=player.angle))>0) - self.cost*indicator_function(action==FIRE))
        # print('State', state)
        # print('Environment', environment)
        # print('Action', action)       
        return player.a*indicator_function(abs(action - indicator_function(environment['angle']!=player.angle))>0) - self.cost*indicator_function(action==FIRE)
    
    def get_reward(self, action):
        """
        Get the reward for the current state.
        """
        return self.utility_function(self.state, action, self.neuron, self.environment)
    

    def custom_reset(self):
        equilibria = self.anonymous.pure_nash()
        self.equilibrium = equilibria[np.random.choice(len(self.anonymous.pure_nash()))]


    

