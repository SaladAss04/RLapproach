from typing import Optional
import numpy as np
import gymnasium as gym

import pygame
import math

from src.env.environment import DummyEnv

def equals(dict1, dict2, tolerance): # Checks whether we need to terminate as we're close enough to the target? -Alex
    assert dict1.keys() == dict2.keys()
    for k in dict1.keys():
        if (dict1[k] < dict2[k] - tolerance[k]).any() or (dict1[k] > dict2[k] + tolerance[k]).any():
            return False
    return True


class ATCplanning(DummyEnv):
    def __init__(self, size: int = 10000, max_acc: np.float16 = 3, max_speed: np.float16 = 300, tolerance = None):
        super().__init__(size=size, max_acc=max_acc, max_speed=max_speed, tolerance=tolerance)

        # Define constants first
        self.max_timesteps = 1000 
        self.max_turn_rate = 10  # Move this before action space definition
        self.min_altitude = 3000

        # Add observation space definition
        self.observation_space = gym.spaces.Dict({
            'agent': gym.spaces.Dict({
                'position': gym.spaces.Box(low=0, high=size, shape=(2,), dtype=np.float32),
                'heading': gym.spaces.Box(low=0, high=360, shape=(1,), dtype=np.float32),
                'altitude': gym.spaces.Box(low=self.min_altitude, high=15000, shape=(1,), dtype=np.float32),
                'speed': gym.spaces.Box(low=-max_speed, high=max_speed, shape=(2,), dtype=np.float32)
            }),
            'target': gym.spaces.Dict({
                'position': gym.spaces.Box(low=0, high=size, shape=(2,), dtype=np.float32),
                'heading': gym.spaces.Box(low=0, high=360, shape=(1,), dtype=np.float32),
                'altitude': gym.spaces.Box(low=self.min_altitude, high=15000, shape=(1,), dtype=np.float32),
                'speed': gym.spaces.Box(low=-max_speed, high=max_speed, shape=(2,), dtype=np.float32)
            })
        })
        
        # Add action space definition
        self.action_space = gym.spaces.Dict({
            'turn': gym.spaces.Box(low=-self.max_turn_rate, high=self.max_turn_rate, shape=(1,), dtype=np.float32),
            'change_horizontal_acceleration': gym.spaces.Box(low=-max_acc, high=max_acc, shape=(1,), dtype=np.float32),
            'change_vertical_acceleration': gym.spaces.Box(low=-max_acc, high=max_acc, shape=(1,), dtype=np.float32)
        })

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

        # Load and prepare the aircraft sprite
        self.sprite_original = None  # Will be loaded when rendering starts
        self.sprite_size = (30, 40)  # Desired sprite size


    def step(self, action):
        # Ensure action values are in the correct format
        action = {
            'turn': np.array([action['turn']], dtype=np.float32),
            'change_horizontal_acceleration': np.array([action['change_horizontal_acceleration']], dtype=np.float32),
            'change_vertical_acceleration': np.array([action['change_vertical_acceleration']], dtype=np.float32)
        }
        
        # Original step code...
        self.timestep += 1
        self._update_agent_state(action)
        
        observation = self._get_obs()
        info = self._get_info()
        reward = self._calculate_reward(observation, info)
        
        terminated = equals(self._agent_state, self._target_state, self.tolerance)
        truncated = self.timestep >= self.max_timesteps

        # Ensure observation matches space definition
        observation = {
            'agent': {
                'position': self._agent_state['position'].astype(np.float32),
                'heading': np.array([self._agent_state['heading']], dtype=np.float32),
                'altitude': np.array([self._agent_state['altitude']], dtype=np.float32),
                'speed': self._agent_state['speed'].astype(np.float32)
            },
            'target': {
                'position': self._target_state['position'].astype(np.float32),
                'heading': np.array([self._target_state['heading']], dtype=np.float32),
                'altitude': np.array([self._target_state['altitude']], dtype=np.float32),
                'speed': self._target_state['speed'].astype(np.float32)
            }
        }
        
        return observation, reward, terminated, truncated, info

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        observation, info = super().reset(seed=seed)
        self.timestep = 0
        self.trajectory = []
        
        # Ensure observation matches space definition
        observation = {
            'agent': {
                'position': self._agent_state['position'].astype(np.float32),
                'heading': np.array([self._agent_state['heading']], dtype=np.float32),
                'altitude': np.array([self._agent_state['altitude']], dtype=np.float32),
                'speed': self._agent_state['speed'].astype(np.float32)
            },
            'target': {
                'position': self._target_state['position'].astype(np.float32),
                'heading': np.array([self._target_state['heading']], dtype=np.float32),
                'altitude': np.array([self._target_state['altitude']], dtype=np.float32),
                'speed': self._target_state['speed'].astype(np.float32)
            }
        }
        
        return observation, info
        

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
    
    def _get_info(self):
        """Calculate and return info dictionary with metrics needed for reward calculation"""
        # Calculate distance
        distance = np.linalg.norm(self._agent_state['position'] - self._target_state['position'])
        
        # Calculate altitude difference
        altitude_difference = abs(float(self._agent_state['altitude']) - float(self._target_state['altitude']))
        
        # Calculate heading difference (considering circular nature of angles)
        heading_difference = min(
            (float(self._agent_state['heading']) - float(self._target_state['heading'])) % 360,
            (float(self._target_state['heading']) - float(self._agent_state['heading'])) % 360
        )
        
        # Calculate speed difference
        speed_difference = np.linalg.norm(self._agent_state['speed'] - self._target_state['speed'])
        
        return {
            'distance': float(distance),
            'altitude_difference': float(altitude_difference),
            'heading_difference': float(heading_difference),
            'speed_difference': float(speed_difference)
        }

    def _calculate_reward(self, observation, info):
        # Normalize distances to [0,1] range
        max_possible_distance = np.sqrt(2) * self.size
        normalized_distance = info['distance'] / max_possible_distance
        normalized_altitude = info['altitude_difference'] / (15000 - self.min_altitude)
        normalized_speed = info['speed_difference'] / (2 * self.max_speed)

        # Base reward focusing on position and altitude
        reward = -0.1  # Small time penalty to encourage efficiency
        reward += 2.0 * (1 - normalized_distance)
        reward += 1.0 * (1 - normalized_altitude)

        # Only consider heading when close to target
        close_to_target = normalized_distance < 0.1  # Within 10% of max distance
        if close_to_target:
            normalized_heading = info['heading_difference'] / 360.0
            reward += 1.0 * (1 - normalized_heading)

        # Terminal conditions
        if equals(self._agent_state, self._target_state, self.tolerance):
            reward += 10.0

        if self._agent_state["altitude"] < self.min_altitude:
            reward -= 10.0

        # Add velocity alignment bonus
        desired_direction = self._target_state["position"] - self._agent_state["position"]
        if np.linalg.norm(desired_direction) > 1e-8:  # Avoid division by zero
            desired_direction = desired_direction / np.linalg.norm(desired_direction)
            current_velocity = self._agent_state["speed"]
            if np.linalg.norm(current_velocity) > 1e-8:
                current_direction = current_velocity / np.linalg.norm(current_velocity)
                alignment = np.dot(desired_direction, current_direction)
                reward += 0.1 * alignment  # Small bonus for moving in right direction

        return float(reward)  # Ensure reward is a float

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
            
    def _load_sprite(self):
        """Load and prepare the sprite image"""
        if self.sprite_original is None:
            try:
                # Load the sprite image
                sprite_path = "src/graphics/sprite.png"
                sprite = pygame.image.load(sprite_path)
                # Convert to the format matching the display
                sprite = sprite.convert_alpha()
                # Scale to desired size
                self.sprite_original = pygame.transform.scale(sprite, self.sprite_size)
            except pygame.error as e:
                print(f"Could not load sprite image: {e}")
                # Create a fallback triangle shape
                self.sprite_original = pygame.Surface(self.sprite_size, pygame.SRCALPHA)
                pygame.draw.polygon(self.sprite_original, (255, 255, 255),
                                 [(15, 0), (30, 40), (0, 40)])
                
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
        """Draw aircraft using the sprite image"""
        # Ensure sprite is loaded
        if self.sprite_original is None:
            self._load_sprite()

        # Create a copy of the sprite to color
        sprite = self.sprite_original.copy()
        
        # Apply color tinting to the sprite
        colored_sprite = pygame.Surface(sprite.get_size(), pygame.SRCALPHA)
        colored_sprite.fill(color)
        sprite.blit(colored_sprite, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
        
        # Calculate heading in radians (same as the line)
        #heading_rad = math.radians(heading)
        
        # Draw heading indicator line
        #end_pos = (pos[0] + self.sprite_size[0] * math.cos(heading_rad),
                #pos[1] - self.sprite_size[0] * math.sin(heading_rad))
        #pygame.draw.line(self.screen, color, pos, end_pos, 2)
        
        # Convert the radian heading to degrees for sprite rotation
        # We use math.degrees() to ensure consistent conversion
        # The negative is because pygame rotation is clockwise
        #rotation_degrees = math.degrees(heading_rad) - 90
        rotated_sprite = pygame.transform.rotate(sprite, heading - 90)
        
        # Get the rect for positioning
        sprite_rect = rotated_sprite.get_rect(center=pos)
        
        # Draw the sprite
        self.screen.blit(rotated_sprite, sprite_rect)
        
        # Draw altitude indicator circle
        #pygame.draw.circle(self.screen, color, pos, self.sprite_size[0]//2, 1)

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
