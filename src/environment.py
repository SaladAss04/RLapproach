from typing import Optional
import numpy as np
import gymnasium as gym

def equals(dict1, dict2, tolerance):
    assert dict1.keys() == dict2.keys()
    for k in dict1.keys():
        if dict1[k] < dict2[k] - tolerance[k] or dict1[k] > dict2[k] + tolerance[k]:
            return False
    return True

class DummyEnv(gym.Env):
    '''
    This is a placeholder environment containing all state, observation and action fields but no transition.
    Feel free to inherit this class when you implement the actual environment.
    '''
    def __init__(self, size: int = 10000, max_acc: np.float16 = 3, max_speed: np.float16 = 300, tolerance = None):
        # Constants
        self.size = size
        self.max_acc = max_acc
        self.max_speed = max_speed
        if tolerance == None:
            self.tolerance = {
                "position": 100.0,
                "heading":10,
                "altutide": 50.0,
                "speed":20.0
            }
        else:
            self.tolerance = tolerance
        # Define the agent and target location; randomly chosen in `reset` and updated in `step`

        self._agent_state = {
            "position": np.array([-1, -1], dtype=np.float16),
            "heading": 0,
            "altitude":np.float16(8000),
            "speed":np.array([200, 0], dtype=np.float16)
        }
        
        self._target_state = {
            "position": np.array([-1, -1], dtype=np.float16),
            "heading": 135,
            "altitude": np.float16(6000),
            "speed":np.array([200, 0], dtype=np.float16)
        }
            
        self.observation_space = gym.spaces.Dict(
            {
                "position": gym.spaces.Box(0, size - 1, shape=(2,), dtype=np.float16),
                "target": gym.spaces.Box(0, size - 1, shape=(2,), dtype=np.float16),
                "heading": gym.spaces.Box(0, 360, shape=(1,), dtype=int),
                "altitude": gym.spaces.Box(0, 15000, shape=(1,), dtype=np.float16),
                "speed": gym.spaces.Box(low=np.array([0.0, -self.max_speed]), high=np.array([self.max_speed, self.max_speed]), dtype=np.float16)
            }
        )

        # We have 4 actions, corresponding to "right", "up", "left", "down"
        self.action_space = gym.spaces.Dict(
            {
                "turn": gym.spaces.Box(-15, 15, shape=(1,), dtype=int),
                "change_horizontal_acceleration": gym.spaces.Box(-self.max_acc, self.max_acc, shape=(1,), dtype=np.float16),
                "change_vertical_acceleration": gym.spaces.Box(-self.max_acc, self.max_acc, shape=(1,), dtype=np.float16)
            }
        )
    
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
        rng = np.random.default_rng()
        
        # Choose the agent's location uniformly at random
        self._agent_state = {
            "position": np.array(rng.uniform(0, self.size, size=2, dtype=np.float16)),
            "heading": np.random.random_integers(0, 360, size=1, dtype=int),
            "altitude": rng.uniform(10000, 15000, size=1, dtype=np.float16),
            "speed": np.array([np.random.random_integers(100, self.max_speed, size=1),
                               0], dtype=np.float16)
        }  
        
        # We will sample the target's location randomly until it does not coincide with the agent's location
        self._target_state = self._agent_state
        while np.array_equal(self._target_state["position"], self._agent_state["position"]):
            self._target_state = {
            "position": np.array(rng.uniform(0, self.size, size=2, dtype=np.float16)),
            "heading": np.random.random_integers(0, 360, size=1, dtype=int),
            "altitude": rng.uniform(5000, 10000, size=1, dtype=np.float16),
            "speed": np.array([np.random.random_integers(50, 200, size=1),
                               0], dtype=np.float16)
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
        acc = np.array(action['change_horizontal_acceleration'], action['change_vertical_acceleration'], dtype=np.float16)
        speed = self._agent_state["speed"]
        degree = self._agent_state["heading"]
        acc = [(lambda x: 0 if speed[x] > self.max_speed else acc[x])(y) for y in [0,1]]
        
        norm_v = np.linalg.norm(acc, ord=2)
        if norm_v > self.max_acc:
            acc = acc * (self.max_acc / norm_v)
        
        _new_state = {
            "speed": np.array([speed[x] + acc[x] for x in [0,1]], dtype=np.float16),
            "altitude": self._agent_state["altitude"] + speed[1],
            "heading": (degree + action["turn"]) % 360,
            "position": np.clip(self._agent_state["position"] + np.array(speed[0] * np.cos(np.radians(degree)),
                                                                 speed[0] * np.sin(np.radians(degree))),
                                0, self.size-1)
        }
        self._agent_state = _new_state
                # An environment is completed if and only if the agent has reached the target
        terminated = equals(self._agent_state, self._target_state, self.tolerance)
        truncated = False
        reward = 1 if terminated else 0  # the agent is only reached at the end of the episode
        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, truncated, info
        