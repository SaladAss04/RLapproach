from typing import Optional
import numpy as np
import gymnasium as gym

from .environment import DummyEnv

def equals(dict1, dict2, tolerance):
    assert dict1.keys() == dict2.keys()
    for k in dict1.keys():
        if (dict1[k] < dict2[k] - tolerance[k]).any() or (dict1[k] > dict2[k] + tolerance[k]).any():
            return False
    return True

class ATCplanning(DummyEnv):
    def __init__(self, size: int = 10000, max_acc: np.float16 = 3, max_speed: np.float16 = 300, tolerance = None):

        super().__init__(size=size, max_acc=max_acc, max_speed=max_speed, tolerance=tolerance)

        self.max_timesteps = 1000 

        #additional parameters
        self.max_turn_rate = 10 #possible turn rate per step
        self.min_altitude = 3000
        self.previous_distance = None
    
    def _get_info(self):
        return {
            "distance": np.linalg.norm(self._agent_state["position"] - self._target_state["position"], ord=1),
            "heading_difference": min(np.abs(self._agent_state["heading"] - self._target_state["heading"]),
                                      360 - np.abs(self._agent_state["heading"] - self._target_state["heading"])),
            "altitude_difference": np.abs(self._agent_state["altitude"] - self._target_state["altitude"]),
            "speed_difference": np.linalg.norm(self._agent_state["speed"] - self._target_state["speed"], ord=1),
            "total_reward": self.episodic_reward,
            "total_steps": self.episodic_step
        }

    
     
    def step(self, action):
        """
        implement environment dynamics
        """

        self.episodic_step += 1
        self._update_agent_state(action)

        observation = self._get_obs()
        info = self._get_info()

        reward = self._calculate_reward(observation, info)
        self.episodic_reward += reward

        terminated = equals(self._agent_state, self._target_state, self.tolerance)
        truncated = self.episodic_step >= self.max_timesteps

        if terminated or truncated:
            self.episodic_step = 0
            self.episodic_reward = 0

        return observation, reward, terminated, truncated, info
        

    def _update_agent_state(self, action):
        """
        physics of state updates, actions {'turn', 'h_acc', 'v_acc'}
        """         

        action = np.asarray(action).flatten()       

        #limit possible actions to realistic airplane turn rate
        turn = float(np.clip(action[0], -self.max_turn_rate, self.max_turn_rate))
        new_heading = float(self._agent_state["heading"] + float(turn)) % 360 #measured in degrees

        x_acc = float(np.clip(action[1], -self.max_acc, self.max_acc))
        y_acc = float(np.clip(action[2], -self.max_acc, self.max_acc))

        curr_speed = self._agent_state["speed"]
        new_speed = np.clip(curr_speed + np.array([x_acc, y_acc]), 
                            [-self.max_speed, -self.max_speed],
                            [self.max_speed, self.max_speed])

        heading_rad = np.radians(new_heading)
        dx = new_speed[0] * np.cos(heading_rad)
        dy = new_speed[0] * np.sin(heading_rad)

        new_position = np.clip(self._agent_state["position"] + np.array([dx, dy]), 0, self.size-1)
        new_altitude = float(np.clip(self._agent_state["altitude"] + new_speed[1], self.min_altitude, 15000)) #add min. and max. altitude?

        self._agent_state.update({
                                     "position": new_position,
                                     "heading": np.array([new_heading]),
                                     "altitude":np.array([new_altitude]),
                                     "speed": new_speed
                                 })
    
    def _calculate_reward(self, observation, info):
        """
        calculate reward 
        """
        #penalize each timestep (shortest path, faster solutions)
        reward = -0.01

        #consider previous distance??
        if self.previous_distance is None:
            self.previous_distance = info['distance']
            distance_improvement = 0
        else:
            distance_improvement = self.previous_distance - info['distance']
            self.previous_distance = info['distance']
            
        reward += 0.1 * distance_improvement

        reward += 0.005 * (1 - info['heading_difference']/180)
        reward += 0.005 * (1 - info['altitude_difference']/12000)
        reward += 0.005 * (1 - info['speed_difference']/(2 * self.max_speed))
        
        #scale rewards to be smaller than step penalty

        #CHANGE: scaled down by 1/100
        #distance_reward = float(-0.0001 * info['distance']) 
        #heading_reward = -0.0001 * float(info['heading_difference']) 
        #altitude_reward = -0.0001 * float(info['altitude_difference']) 
        #speed_reward = -0.0001 * float(info['speed_difference'])

        #reward += distance_reward + heading_reward + altitude_reward + speed_reward

        #terminal reward/penalty
        #CHANGE: from 100
        if equals(self._agent_state, self._target_state, self.tolerance):
            reward += 1.0

        if self._agent_state["altitude"] < self.min_altitude:
            self.previous_distance = None
            reward -= 1.0

        return reward

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        observation, info = super().reset(seed=seed)
        self.episodic_reward = 0
        self.episodic_step = 0
        self.previous_distance = None

        return observation, info

    """
    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
        self.trajectory = []
    """
