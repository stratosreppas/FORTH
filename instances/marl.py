import csv
from datetime import datetime
from games.markovian.MARL import MARL
from pettingzoo.utils import wrappers, agent_selector
import gymnasium as gym
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity 

from gymnasium.spaces import Dict as GymDict, Box, Discrete

from games.player import Player
from global_utils import indicator_function

# Observations - Empty Observation
EMPTY = -1

# Actions
NOT_FIRE = 0
FIRE = 1

# Angles
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

class Neuron(Player):

    def __init__(self, name='Neuron', actions=[NOT_FIRE, FIRE], utility_params={'cost': 0.1, 'alpha': 0.5}):
        super().__init__(name, actions, utility_params)
        self.angle = np.random.randint(ANGLE_0, ANGLE_337_5)

    def utility_function(self, action, angle):
        a = self.utility_params['alpha']
        c = self.utility_params['cost']
        or_pref = self.angle*22.5
        angle = angle*22.5
        cos_sim = np.cos(np.deg2rad(or_pref - angle))
        return a**(indicator_function(cos_sim<0))*abs(cos_sim) - c*(indicator_function(action==FIRE))

class MARL(MARL):

    def __init__(self, name='marl', num_neurons=40, cost=0.1, alpha=np.random.rand()):

        utility_params = {'cost': cost, 'alpha': alpha}

        players = [Neuron(name=f'L4Neuron_{i}', utility_params=utility_params) for i in range(num_neurons)] + [Neuron(name='L23Neuron', utility_params=utility_params)]
        
        super().__init__(name, players)

        self.environment = {}
        self.possible_agents = [player.name for player in players]
        self.segment = 0
        self.num_moves = 0
        
        self.angle = None

    def step(self, action):
        """
        Perform a single step in the environment.
        
        Args:
            action: The action to be taken by the agent.
        Returns:
            None
        Raises:
            None
        """
        agent = self.agent_selection

        # Handle the agent being masked
        if (self.terminations[agent] or self.truncations[agent]):
            self._was_dead_step(action)
        
        else:        
            self.timestep += 1
            self.state[agent] = action

            # Collect the reward if this is the last L4 agent or the L23 agent
            if(agent=='L23Neuron'):
                self.num_moves += 1
                # if the L23 agent is stepping, collect the rewards and reveal them to all the agents
                self.observations = {agent: self.observe(agent) for agent in self.agents}
                self.rewards = {agent: self.get_reward(agent) for agent in self.agents}
                
                # print(self.num_moves, self.max_cycles, self.num_moves % self.max_cycles == 0)

                self.truncations = {
                    agent: self.num_moves % self.max_cycles == 0 for agent in self.agents
                }
                
                self.done = self.truncations or self.terminations
                # print(self.done)
                
                # 
                self.save_obs_to_csv()
                

            
            else:
                # Empty state for the neurons that have not played yet
                for i in range(self.agent_name_mapping[agent]+1, len(self.agents)):
                    self.state[self.agents[i]] = EMPTY
                # no rewards are allocated until the L23 agent steps
                self._clear_rewards()

            self._cumulative_rewards[agent] = 0
            self.agent_selection = self.agent_selector.next()
            self._accumulate_rewards()
            
    def custom_reset(self):
        """
        Custom reset function for the environment. 
        It is used in case the environment needs to be reset in a specific way.
        
        Args:
            None
        Returns:
            None
        Raises:
            None
        """
        print('resetting')
        self.segment += 1
        self.angle = np.random.randint(ANGLE_0, ANGLE_337_5)

    def get_observation_space(self):
        """
        Returns the observation space for the multi-agent reinforcement learning environment.
        Returns:
            GymDict: The observation space, which is a dictionary containing the following keys:
                - "l4_obs": Represents the observation space for the l4 agents. It is a discrete space with a maximum value of `self.max_num_agents + 1`.
                - "l23_obs": Represents the observation space for the l23 agents. It is a discrete space with a maximum value of `ANGLE_337_5 + 1`.
        """
        return GymDict(
            {
            "l4_obs": Box(low=0, high=self.max_num_agents+1, shape=(1,), dtype=np.float32), 
            "l23_obs": Box(low=0, high=ANGLE_337_5+1, shape=(1,), dtype=np.float32)
            }
        )
    
    def get_action_space(self):
        return Discrete(2)
    
    def observation_space(self, agent):
        """
        Returns the observation space for the particular agent.
        
        Parameters
        ----------
        
        agent : str
            The agent for which to return the observation space
            
        Returns
        -------
        
        observation_space : gym.spaces
            The observation space for the agent. 
            
            In this game, it is a Discrete space with the number of neurons in the L4 layer + 1 for the L23 agent, 
            or a Discrete space with the number of angles in the environment for the L23 agent.
        """
        if agent == 'L23Neuron':
            return self.get_observation_space()['l23_obs']
        else:
            return self.get_observation_space()['l4_obs']
    
    def action_space(self, agent):
        return self.get_action_space()
    
    def observe(self, agent):
        """
        Returns an observation of the environment for the particular agent.
        
        Parameters
        ----------
        
        agent : str
            The agent for which to return an observation
        
        Returns
        -------

        observation : int
            The observation for the agent. 
            
            In this game, it is an int, signifying either the angle presented to the L4 layer 
            or the number of cofiring events of the L4 layer, 
            as the observation for the L4 and L23 agents, respectively. 
        
        """
        if agent == 'L23Neuron':
            return self.normalized_obs(sum(self.get_state()[agent] for agent in self.agents), agent) # Anonymity implementation
        else:
            return self.normalized_obs(self.angle, agent) 
    
    def save_obs_to_csv(self):
        file = open(self.obs_csv, 'a+', newline='')
        
        fields = []
        with file:
            fields.append('date')
            fields.append('segment')
            fields.append('steps')
            for agent in self.possible_agents:
                fields.append(agent+'_obs')
                fields.append(agent+'_action')
                fields.append(agent+'_reward')
                
            writer = csv.DictWriter(file, fieldnames=fields)
            
            # Write the titles to the CSV file
            if file.tell() == 0:
                writer.writeheader()
            
            row = {'date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                   'segment': self.segment,
                   'steps': self.timestep
                   }
            for agent in self.possible_agents:
                row[f'{agent}_obs'] = self.observations[agent]
                row[f'{agent}_action'] = self.get_state()[agent]
                row[f'{agent}_reward'] = self.rewards[agent]
            
            writer.writerow(row)
            
            return
    
    def get_reward(self, agent):
        return self.players[self.agent_name_mapping['L23Neuron']].utility_function(self.get_state()[agent], self.observations[agent])