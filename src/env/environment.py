from typing import Optional
import numpy as np
import gymnasium as gym
from gymnasium.envs.registration import register

def equals(dict1, dict2, tolerance):
    assert dict1.keys() == dict2.keys()
    for k in dict1.keys():
        if (dict1[k] < dict2[k] - tolerance[k]).any() or (dict1[k] > dict2[k] + tolerance[k]).any():
            return False
    return True

def crash(s, a, min_s = 100.0, min_a = 300.0):
    return s[0] < min_s or a < min_a

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
            "distance": np.linalg.norm(
                self._agent_state["position"] - self._target_state["position"], ord=1
            ),
            "heading_difference": min(np.abs(self._agent_state["heading"] - self._target_state["heading"]), 360 - np.abs(self._agent_state["heading"] - self._target_state["heading"])),
            "altitude_difference": np.abs(self._agent_state["altitude"] - self._target_state["altitude"]),
            "speed_difference": np.linalg.norm(
                self._agent_state["speed"] - self._target_state["speed"], ord=1
            )
        }
        
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        self.current_step = 0
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
        terminated = (win or lose)
        truncated = self.current_step > self.max_steps
        reward = 1 if win else (-300 if lose else -1)
        observation = self._get_obs()
        info = self._get_info()

        self.current_step += 1
        return observation, reward, terminated, truncated, info
    
    def render(self):
        pass
    
