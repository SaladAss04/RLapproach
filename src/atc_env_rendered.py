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
        self.render_mode = 'human'
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
    def render(self):
        if self.render_mode is None:
            return

        if self.screen is None:
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            pygame.display.set_caption('ATC Planning Environment')
            
            # Create additional fonts for different text sizes
            self.title_font = pygame.font.SysFont('Arial', 24, bold=True)
            self.info_font = pygame.font.SysFont('Arial', 16)
            self.small_font = pygame.font.SysFont('Arial', 12)

        # Update colors with a more professional scheme
        self.colors.update({
            'background': (15, 15, 35),      # Dark blue-gray
            'grid': (30, 30, 50),            # Lighter grid lines
            'agent': (0, 255, 0),            # Green for agent
            'target': (255, 100, 100),       # Soft red for target
            'text': (200, 200, 200),         # Light gray for text
            'altitude_high': (120, 180, 255), # Blue for high altitude
            'altitude_low': (255, 140, 0),    # Orange for low altitude
            'warning': (255, 60, 60),         # Red for warnings
            'trajectory': (100, 100, 255, 128) # Semi-transparent blue
        })
        
        # Clear screen
        self.screen.fill(self.colors['background'])
        
        # Draw grid
        self._draw_grid()
        
        # Draw airspace boundaries and info panel backgrounds
        self._draw_airspace()
        self._draw_info_panels()
        
        # Convert world coordinates to screen coordinates
        def world_to_screen(pos):
            x = self.padding + (pos[0] / self.size) * (self.screen_width - 2 * self.padding)
            # Flip y-axis for traditional coordinate system
            y = self.screen_height - (self.padding + (pos[1] / self.size) * (self.screen_height - 2 * self.padding))
            return (int(x), int(y))
        
        # Draw trajectory with altitude-based coloring
        if len(self.trajectory) > 1:
            for i in range(len(self.trajectory) - 1):
                alt_ratio = (self._agent_state['altitude'] - self.min_altitude) / (15000 - self.min_altitude)
                color = self._interpolate_color(self.colors['altitude_low'], self.colors['altitude_high'], alt_ratio)
                pygame.draw.line(self.screen, color, self.trajectory[i], self.trajectory[i+1], 2)
        
        # Draw aircraft (agent)
        agent_pos = world_to_screen(self._agent_state["position"])
        self.trajectory.append(agent_pos)
        
        # Draw altitude indicator circle
        alt_ratio = (self._agent_state['altitude'] - self.min_altitude) / (15000 - self.min_altitude)
        altitude_color = self._interpolate_color(self.colors['altitude_low'], self.colors['altitude_high'], alt_ratio)
        
        # Draw aircraft symbol
        self._draw_aircraft(agent_pos, self._agent_state["heading"], altitude_color)
        
        # Draw target
        target_pos = world_to_screen(self._target_state["position"])
        self._draw_target(target_pos, self._target_state["heading"])
        
        # Draw information displays
        self._draw_state_info()
        self._draw_navigation_info()
        
        # Update display
        pygame.display.flip()
        
        if self.render_mode == "rgb_array":
            return pygame.surfarray.array3d(self.screen).transpose((1, 0, 2))

    def _draw_grid(self):
        # Draw coordinate grid
        grid_spacing = 50
        for x in range(self.padding, self.screen_width - self.padding, grid_spacing):
            pygame.draw.line(self.screen, self.colors['grid'], (x, self.padding), 
                           (x, self.screen_height - self.padding))
        for y in range(self.padding, self.screen_height - self.padding, grid_spacing):
            pygame.draw.line(self.screen, self.colors['grid'], (self.padding, y), 
                           (self.screen_width - self.padding, y))

    def _draw_airspace(self):
        # Draw minimum altitude boundary
        pygame.draw.rect(self.screen, self.colors['grid'], 
                        (self.padding, self.padding, 
                         self.screen_width - 2*self.padding, 
                         self.screen_height - 2*self.padding), 2)
        
        # Add altitude scale on the side
        scale_width = 30
        scale_height = self.screen_height - 2*self.padding
        scale_x = self.screen_width - self.padding + 10
        
        for i in range(11):
            y = self.padding + (scale_height * (10-i) // 10)
            alt = self.min_altitude + (15000 - self.min_altitude) * i / 10
            alt_text = self.small_font.render(f"{int(alt)}ft", True, self.colors['text'])
            self.screen.blit(alt_text, (scale_x, y))

    def _draw_aircraft(self, pos, heading, color):
        # Draw more detailed aircraft symbol
        heading_rad = math.radians(heading)
        size = 15
        
        # Aircraft shape points relative to center
        points = [
            (0, -size),  # nose
            (size//2, size//2),  # right wing
            (0, size//4),  # body
            (-size//2, size//2),  # left wing
        ]
        
        # Rotate points according to heading
        rotated_points = []
        for x, y in points:
            rx = x * math.cos(heading_rad) - y * math.sin(heading_rad)
            ry = x * math.sin(heading_rad) + y * math.cos(heading_rad)
            rotated_points.append((pos[0] + rx, pos[1] + ry))
        
        # Draw aircraft
        pygame.draw.polygon(self.screen, color, rotated_points)
        
        # Draw altitude indicator circle
        pygame.draw.circle(self.screen, color, pos, size+5, 1)

    def _draw_target(self, pos, heading):
        # Draw target indicator
        size = 20
        pygame.draw.circle(self.screen, self.colors['target'], pos, size, 2)
        heading_rad = math.radians(heading)
        end_pos = (pos[0] + size * math.cos(heading_rad), 
                  pos[1] + size * math.sin(heading_rad))
        pygame.draw.line(self.screen, self.colors['target'], pos, end_pos, 2)

    def _draw_info_panels(self):
        # Aircraft state panel
        self._draw_state_info()
        # Navigation info panel
        self._draw_navigation_info()
        # Draw status warnings if needed
        #self._draw_warnings()

    def _draw_state_info(self):
        # Create information panel
        info_surface = pygame.Surface((250, 150))
        info_surface.fill((30, 30, 50))
        
        # Add title
        title = self.title_font.render("Aircraft State", True, self.colors['text'])
        info_surface.blit(title, (10, 5))
        
        # Add state information
        info_texts = [
            f"Position: ({self._agent_state['position'][0]:.1f}, {self._agent_state['position'][1]:.1f})",
            f"Heading: {self._agent_state['heading']:.1f}°",
            f"Altitude: {self._agent_state['altitude']:.1f} ft",
            f"Speed: {np.linalg.norm(self._agent_state['speed']):.1f} knots",
        ]
        
        for i, text in enumerate(info_texts):
            text_surface = self.info_font.render(text, True, self.colors['text'])
            info_surface.blit(text_surface, (10, 35 + i * 25))
        
        self.screen.blit(info_surface, (10, 10))

    def _draw_navigation_info(self):
        # Create navigation info panel
        nav_surface = pygame.Surface((250, 150))
        nav_surface.fill((30, 30, 50))
        
        # Add title
        title = self.title_font.render("Navigation", True, self.colors['text'])
        nav_surface.blit(title, (10, 5))
        
        # Calculate additional navigation information
        distance_to_target = np.linalg.norm(
            self._agent_state['position'] - self._target_state['position'])
        altitude_difference = self._agent_state['altitude'] - self._target_state['altitude']
        heading_difference = (self._target_state['heading'] - self._agent_state['heading']) % 360
        
        nav_texts = [
            f"Distance: {distance_to_target:.1f} nm",
            f"Alt Diff: {altitude_difference:.1f} ft",
            f"Hdg Diff: {heading_difference:.1f}°",
            f"Time: {self.timestep}/{self.max_timesteps}"
        ]
        
        for i, text in enumerate(nav_texts):
            text_surface = self.info_font.render(text, True, self.colors['text'])
            nav_surface.blit(text_surface, (10, 35 + i * 25))
        
        self.screen.blit(nav_surface, (10, 170))

    @staticmethod
    def _interpolate_color(color1, color2, ratio):
        return tuple(int(c1 + (c2 - c1) * ratio) for c1, c2 in zip(color1, color2))
        
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
