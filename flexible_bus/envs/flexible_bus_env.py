import gym
import numpy as np
# import pybullet as p
import numpy as np
import pandas as pd
from flexible_bus.resources.demand_sim import ridership_cal,get_demand
class FlexibleBusEnv(gym.Env):
    metadata = {'render.modes': ['human']}  
  
    def __init__(self):

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
        self.action_space = gym.spaces.box.Box(
            low=np.array([0, 0],dtype=np.float32),
            high=np.array([1, 1],dtype=np.float32))
        self.observation_space = gym.spaces.box.Box(
            low=np.array([-np.inf, -np.inf, -np.inf, -np.inf, -np.inf],dtype=np.float32),
            high=np.array([np.inf, np.inf, np.inf, np.inf, np.inf],dtype=np.float32))
        self.np_random, _ = gym.utils.seeding.np_random()
        self.timestamp = None
        #self.time = time
        self.start = None
        self.map = pd.read_csv("./flexible_bus/resources/demand/test.csv")
        self.demand_dist = None
        self.tol = 0.1
        
        self.state = None
        self.done = None
        self.reset()
    def step(self, action):
        reward = ridership_cal(action[0], action[1], self.demand_dist,self.tol)
        self.demand_dist = get_demand(1)
        # apply the action to simulator to get the ridership
        
        #self.demand_dist = self.map.iloc[self.timestamp,1:].tolist()
        #calculate the ridership into reward
        observation = self.demand_dist
        
        #
        done = bool(self.timestamp >= 4)
        self.timestamp += 1  
        return observation, reward, done,{}
    def reset(self):
        # self.timestamp = np.random.randint(0, 1200)
        # self.start = self.timestamp
        self.timestamp = 0
        self.demand_dist = get_demand(1)
        return self.demand_dist

    def render(self):
        pass

    def close(self):
        pass
    def seed(self, seed=None): 
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]