from typing import Optional
import numpy as np
import gymnasium as gym

import pygame
import math

from src.environment import DummyEnv

def equals(dict1, dict2, tolerance):
    assert dict1.keys() == dict2.keys()
    for k in dict1.keys():
        if (dict1[k] < dict2[k] - tolerance[k]).any() or (dict1[k] > dict2[k] + tolerance[k]).any():
            return False
    return True

class ATCplanning(DummyEnv):
    def __init__(self, size: int = 10000, max_acc: np.float16 = 3, max_speed: np.float16 = 300, tolerance = None):

        super().__init__(size=size, max_acc=max_acc, max_speed=max_speed, tolerance=tolerance)

        self.timestep = 0 
        self.max_timesteps = 1000 

        #additional parameters
        self.max_turn_rate = 10 #possible turn rate per step
        self.min_altitude = 3000

        #rendering setup
        self.screen_width = 800
        self.screen_height = 600
        self.padding = 50
        self.screen = None
        pygame.font.init()
        self.font = pygame.font.SysFont('Arial', 16)

        self.colors = {
            'background': (50, 50, 50),     # Dark gray
            'agent': (0, 255, 0),           # Green
            'target': (255, 0, 0),          # Red
            'text': (255, 255, 255),        # White
            'trajectory': (100, 100, 255)    # Light blue
        }

        #store trajectory
        self.trajectory = []

    def step(self, action):
        """
        implement environment dynamics
        """

        self.timestep += 1
        self._update_agent_state(action)

        observation = self._get_obs()
        info = self._get_info()

        reward = self._calculate_reward(observation, info)

        terminated = equals(self._agent_state, self._target_state, self.tolerance)
        truncated = self.timestep >= self.max_timesteps

        return observation, reward, terminated, truncated, info
        

    def _update_agent_state(self, action):
        """
        physics of state updates, actions {'turn', 'h_acc', 'v_acc'}
        """                

        #limit possible actions to realistic airplane turn rate
        turn = float(np.clip(action['turn'], -self.max_turn_rate, self.max_turn_rate))
        new_heading = float(self._agent_state["heading"] + float(turn)) % 360 #measured in degrees

        x_acc = float(np.clip(action['change_horizontal_acceleration'], -self.max_acc, self.max_acc))
        y_acc = float(np.clip(action['change_vertical_acceleration'], -self.max_acc, self.max_acc))

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
                                     "heading": new_heading,
                                     "altitude": new_altitude,
                                     "speed": new_speed
                                 })
    
    def _calculate_reward(self, observation, info):
        """
        calculate reward 
        """
        #penalize each timestep (shortest path, faster solutions)
        reward = -0.1

        #scale rewards to be smaller than step penalty
        distance_reward = float(-0.01 * info['distance']) 
        heading_reward = -0.01 * float(info['heading_difference']) 
        altitude_reward = -0.01 * float(info['altitude_difference']) 
        speed_reward = -0.01 * float(info['speed_difference'])

        reward += distance_reward + heading_reward + altitude_reward + speed_reward

        #terminal reward/penalty
        if equals(self._agent_state, self._target_state, self.tolerance):
            reward += 100

        if self._agent_state["altitude"] < self.min_altitude:
            reward -= 100

        return reward

    #temporary render function for basic functionalities
    def render(self, mode='human'):
        if self.screen is None:
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            pygame.display.set_caption('ATC Planning Environment')
            
        # Clear screen
        self.screen.fill(self.colors['background'])
        
        # Convert world coordinates to screen coordinates
        def world_to_screen(pos):
            x = self.padding + (pos[0] / self.size) * (self.screen_width - 2 * self.padding)
            y = self.padding + (pos[1] / self.size) * (self.screen_height - 2 * self.padding)
            return (int(x), int(y))
        
        # Draw trajectory
        if len(self.trajectory) > 1:
            pygame.draw.lines(self.screen, self.colors['trajectory'], False, self.trajectory, 2)
        
        # Draw agent
        agent_pos = world_to_screen(self._agent_state["position"])
        self.trajectory.append(agent_pos)
        pygame.draw.circle(self.screen, self.colors['agent'], agent_pos, 5)
        
        # Draw agent heading indicator
        heading_rad = math.radians(self._agent_state["heading"])
        heading_length = 20
        heading_end = (
            agent_pos[0] + int(heading_length * math.cos(heading_rad)),
            agent_pos[1] + int(heading_length * math.sin(heading_rad))
        )
        pygame.draw.line(self.screen, self.colors['agent'], agent_pos, heading_end, 2)
        
        # Draw target
        target_pos = world_to_screen(self._target_state["position"])
        pygame.draw.circle(self.screen, self.colors['target'], target_pos, 5)
        
        # Draw state information
        info_texts = [
            f"Position: ({self._agent_state['position'][0]:.1f}, {self._agent_state['position'][1]:.1f})",
            f"Heading: {self._agent_state['heading']:.1f}°",
            f"Altitude: {self._agent_state['altitude']:.1f} ft",
            f"Speed: {np.linalg.norm(self._agent_state['speed']):.1f} knots"
        ]
        
        for i, text in enumerate(info_texts):
            text_surface = self.font.render(text, True, self.colors['text'])
            self.screen.blit(text_surface, (10, 10 + i * 20))
        
        if mode == 'human':
            pygame.display.flip()
            
        elif mode == 'rgb_array':
            screen_data = pygame.surfarray.array3d(self.screen)
            return screen_data.transpose((1, 0, 2))

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        observation, info = super().reset(seed=seed)
        self.timestep = 0
        self.trajectory = []

        return observation, info

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
            self.screen = None
        self.trajectory = []
