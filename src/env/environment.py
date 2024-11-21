from typing import Optional
import numpy as np
import gymnasium as gym
from gymnasium.envs.registration import register
import math

def equals(dict1, dict2, tolerance):
    assert dict1.keys() == dict2.keys()
    for k in dict1.keys():
        if (dict1[k] < dict2[k] - tolerance[k]).any() or (dict1[k] > dict2[k] + tolerance[k]).any():
            return False
    return True

def crash(s, a, min_s = 100.0, min_a = 300.0):
    return s[0] < min_s or a[0] < min_a

class DummyEnv(gym.Env):
    '''
    This is a placeholder environment containing all state, observation and action fields but no transition.
    Feel free to inherit this class when you implement the actual environment.
    '''
    def __init__(self, size: int = 10000, max_acc: float= 3, max_speed: float= 300, max_steps = 300, tolerance = None):
        # Constants
        self.size = size
        self.max_acc = max_acc
        self.max_speed = max_speed
        self.max_steps = max_steps
        self.episodic_reward = 0
        self.episodic_step = 0
        
        if tolerance == None:
            self.tolerance = {
                "position": 100.0,
                "heading":10,
                "altitude": 50.0,
                "speed":20.0
            }
        else:
            self.tolerance = tolerance
        # Define the agent and target location; randomly chosen in `reset` and updated in `step`
        self.rng = np.random.default_rng()
        
        self._agent_state = {
            "position": np.array([-1, -1], dtype=float),
            "heading": 0,
            "altitude":float(8000),
            "speed":np.array([200, 0], dtype=float)
        }
        
        self._target_state = {
            "position": np.array([-1, -1], dtype=float),
            "heading": 135,
            "altitude": float(6000),
            "speed":np.array([200, 0], dtype=float)
        }
            
        self.observation_space = gym.spaces.Dict(
            {   
                "agent":gym.spaces.Dict({
                    "speed": gym.spaces.Box(low=np.array([0.0, -self.max_speed]), high=np.array([self.max_speed, self.max_speed]), dtype=float),
                    "altitude": gym.spaces.Box(0, 15000, shape=(1,), dtype=float),
                    "heading": gym.spaces.Box(0, 360, shape=(1,), dtype=float),
                    "position": gym.spaces.Box(0, size - 1, shape=(2,), dtype=float)
                }),
                "target": gym.spaces.Dict({
                    "speed": gym.spaces.Box(low=np.array([0.0, -self.max_speed]), high=np.array([self.max_speed, self.max_speed]), dtype=float),
                    "altitude": gym.spaces.Box(0, 15000, shape=(1,), dtype=float),
                    "heading": gym.spaces.Box(0, 360, shape=(1,), dtype=float),
                    "position": gym.spaces.Box(0, size - 1, shape=(2,), dtype=float)
                })
            }
        )
        '''
        self.action_space = gym.spaces.Dict(
            {
                "turn": gym.spaces.Box(-15, 15, shape=(1,), dtype=int),
                "h_acc": gym.spaces.Box(-self.max_acc, self.max_acc, shape=(1,), dtype=float),
                "v_acc": gym.spaces.Box(-self.max_acc, self.max_acc, shape=(1,), dtype=float)
            }
        )
        '''
        self.action_space = gym.spaces.Box(low=np.array([-15.0, -self.max_acc, -self.max_acc]), high=np.array([15.0, self.max_acc, self.max_acc]), dtype=float)
    
    def _get_obs(self):
        return {"agent": self._agent_state, "target": self._target_state}
    
    def _get_info(self):
        return {
            "total_reward": self.episodic_reward,
            "total_steps": self.episodic_step
        }
        
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        self.episodic_step = 0
        self.episodic_reward = 0
        # Choose the agent's location uniformly at random
        self._agent_state = {
            "position": np.array(self.rng.uniform(0, self.size, size=2)),
            "heading": np.random.random_integers(0, 360, size=1),
            "altitude": self.rng.uniform(10000, 15000, size=1),
            "speed": np.array([np.random.uniform(100, self.max_speed, size=1)[0],
                               float(0)])
        }  
        
        # We will sample the target's location randomly until it does not coincide with the agent's location
        self._target_state = self._agent_state
        while np.array_equal(self._target_state["position"], self._agent_state["position"]):
            self._target_state = {
            "position": np.array(self.rng.uniform(0, self.size, size=2)),
            "heading": np.random.random_integers(0, 360, size=1),
            "altitude": self.rng.uniform(5000, 10000, size=1),
            "speed": np.array([np.random.random_integers(50, 200, size=1)[0],
                               float(0)])
            }  

        observation = self._get_obs()
        info = self._get_info()

        return observation, info
    
    def step(self, action):
        '''
        The input action should be a dict {'turn', 'h_acc', 'v_acc'}.
        If norm of accelerations exceed the limit, they will be normalized.
        If speed is reaching maximum, the corresponding acc will be set to 0.
        '''
        acc = np.array([action[1], action[2]], dtype=float)
        speed = self._agent_state["speed"]
        degree = self._agent_state["heading"]
        acc = [(lambda x: 0 if speed[x] > self.max_speed else acc[x])(y) for y in [0,1]]
        
        norm_v = np.linalg.norm(acc, ord=2)
        if norm_v > self.max_acc:
            acc = [acc[x] * (self.max_acc / norm_v) for x in [0, 1]]
        
        _new_state = {
            "speed": np.array([speed[x] + acc[x] for x in [0,1]], dtype=float),
            "altitude": self._agent_state["altitude"] + speed[1],
            "heading": (degree + action[0]) % 360,
            "position": np.clip(self._agent_state["position"].reshape((-1,)) + np.array([speed[0] * np.cos(np.radians(degree)),
                                                                 speed[0] * np.sin(np.radians(degree))]).reshape((-1,)),
                                0, self.size-1)
        }
        self._agent_state = _new_state
        win = equals(self._agent_state, self._target_state, self.tolerance)
        lose = crash(self._agent_state['speed'], self._agent_state['altitude'])
        terminated = win or lose
        truncated = self.episodic_step > self.max_steps
        reward = 500 if win else (-300 if lose else -1)
        observation = self._get_obs()
        info = self._get_info()

        self.episodic_step += 1
        self.episodic_reward += reward
        if terminated or truncated:
            self.episodic_step = 0
            self.episodic_reward = 0
        return observation, reward, terminated, truncated, info
    
    def render(self):
        pass
    
def calculate_heading(agent, target):    
    x1, y1 = agent
    x2, y2 = target
    dx = x2 - x1
    dy = y2 - y1
    angle_rad = math.atan2(dy, dx)

    angle_deg = math.degrees(angle_rad)
    if angle_deg < 0:
        angle_deg += 360

    return round(angle_deg / 10) * 10
    
def calculate_position(agent, heading, v):
    rad = math.radians(heading)
    return np.array([agent[0] + v * math.cos(rad), 
                     agent[1] + v * math.sin(rad)])
    
def arrive(agent, target, tolerance):
    return np.linalg.norm(target - agent) < tolerance
    
class DiscreteApproach(gym.Env):
    def __init__(self, tolerance = None, max_steps = 300):
        self.size = 100
        self.speed = 1
        self.slight_turn = 10
        self.hard_turn = 30
        self.num_stat_obs = 3
        self.num_mot_obs = 2
        self.radius = 10 #the radius within which we consider an obstacle an intruder
        
        self.max_steps = max_steps
        self.episodic_reward = 0
        self.episodic_step = 0
        if tolerance == None:
            #tolerable in a circle with radius 2
            self.tolerance = 2
        else:
            self.tolerance = tolerance
        self._agent_state = None
        self._target_state = None
        self._obstacles = None
        '''
        obs space:
        A. distance to intruder, angle to intruder, relative heading to intruder;
            considering multiple intruders.
            first static ones, then motional ones.
        
        B. distance to target, angle to target.
            target considered as a static obstacle.
        
        distances are in float, while angles are in 10-degrees.
        '''
        low = np.tile(np.array([0.0, 0, 0]), (self.num_mot_obs + self.num_stat_obs + 1, 1))
        high = np.tile(np.array([self.size, 35, 35]), (self.num_mot_obs + self.num_stat_obs + 1, 1))
        self.observation_space = gym.spaces.Box(
            low = low,
            high = high,
            dtype = np.float32
        )
        self.observation_space[-1][-1] = -1 #means static
        '''
        Action Space:
        Stay, Slight Left, Slight Right, Hard Left, Hard Right.
        
        t.b.d: slight turns are 10 degrees, while hard turns are 30 degrees.
        '''
        self.action_space = gym.spaces.Discrete(5)
        
    def _get_obs(self):
        pos, heading = self._agent_state["position"], self._agent_state["heading"]
        target = self._target_state
        obs = self.observation_space

        for i, x in enumerate(self._obstacles["static"]):
            obs[i][0] = np.linalg.norm(pos - x)
            obs[i][1] = calculate_heading(pos, x)
            obs[i][2] = -1
            assert i < obs.shape[0] - 1
        for i, x in enumerate(self._obstacles["motional"]):
            raise NotImplementedError("Motional obstacles not implemented, but observing.")

        obs[-1][0] = np.linalg.norm(pos - target)
        obs[-1][1] = calculate_heading(pos, target)
        obs[-1][2] = -1

        self.observation_space = obs
        return self.observation_space
    
    def _get_info(self):
        return {
            "total_reward": self.episodic_reward,
            "total_steps": self.episodic_step,
            "agent_state": self._agent_state,
            "target_state": self._target_state
        }
    
    def reset(self, seed: Optional[int] = None, motional_obstacles = False):
        super().reset(seed=seed)
        self.episodic_step = 0
        self.episodic_reward = 0
        self._target_state = np.array(self.rng.uniform(0, self.size, size=2))

        self._agent_state = {
            "position": np.array(self.rng.uniform(0, self.size, size=2)),
            "heading": calculate_heading(self._agent_state["position"], self._target_state["position"]),
        }
        #initialize obstacles related information
        if not motional_obstacles:
            self._obstacles = {
                "static":[np.array(self.rng.uniform(0, self.size, size=2)) for _ in range(self.num_stat_obs)],
                "motional": None
            }
        else:
            raise NotImplementedError("Motional obstacles not implemented, but initializing.")

        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def step(self, action, alpha = 1.0, beta = 0.5, gamma = 4.0):
        '''
        Action Space:
        Stay, Slight Left, Slight Right, Hard Left, Hard Right.
        
        t.b.d: slight turns are 10 degrees, while hard turns are 30 degrees.
        '''
        pos, heading = self._agent_state["position"], self._agent_state["heading"]
        angle_change = [0, self.slight_turn, -self.slight_turn,
                        self.hard_turn, -self.hard_turn]
        _new_state = {
            "position": calculate_position(pos, heading, self.speed),
            "heading": heading + angle_change[action]
        }
        self._agent_state = _new_state
        lose = [arrive(_new_state["position"], x, self.tolerance)
                for x in self._obstacles["static"].extend(self._obstacles["motional"])].any()
        win = arrive(_new_state["position"],
                           self._target_state["position"], self.tolerance)
        terminated = win or lose
        truncated = self.episodic_step > self.max_steps
        
        obs = self._get_obs()
        reward_target = alpha * (self.size - obs[-2]) + beta * (360 - obs[-1])
        penalty_obstacle = 0
        for i, x in enumerate(obs):
            if i == obs.shape[0] - 1:
                break
            dis = x[0] if x[0] < self.radius else 0
            penalty_obstacle += self.radius * self.radius - dis * dis
        penalty_obstacle *= gamma 
        reward_terminate = 500 if win else(-500 if lose else 0)

        reward = reward_target + penalty_obstacle + reward_terminate
        self.episodic_step += 1
        self.episodic_reward += reward
        if terminated or truncated:
            self.episodic_step = 0
            self.episodic_reward = 0
        
        info = self._get_info()
        
        return obs, reward, terminated, truncated, info