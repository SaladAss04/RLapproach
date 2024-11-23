from typing import Optional
import numpy as np
import gymnasium as gym
from gymnasium.envs.registration import register
import math
import pygame

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
        self.episodic_step += 1
        self.episodic_reward += reward
        info = self._get_info()
        '''
        if terminated or truncated:
            self.episodic_step = 0
            self.episodic_reward = 0
        '''
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
    
class DiscreteApproach(DummyEnv):
    def __init__(self, tolerance = None, max_steps = 300):
        self.size = 100
        self.speed = 1
        self.slight_turn = 10
        self.hard_turn = 30
        self.num_stat_obs = 3
        self.num_mot_obs = 0
        self.radius = 4 #the radius within which we consider an obstacle an intruder
        self.rng = np.random.default_rng()
        self.max_speed = 300
        
        self.max_steps = max_steps
        self.episodic_reward = 0
        self.episodic_step = 0
        if tolerance == None:
            #tolerable in a circle with radius 2
            self.tolerance = 2
        else:
            self.tolerance = tolerance
        self._agent_state = None
        #self._target_state = None
        self._obstacles = None
        self.episodic_info = {
            "total_reward": self.episodic_reward,
            "total_steps": self.episodic_step
        }
        
        #rendering
        self.screen_width = 800
        self.screen_height = 600
        self.padding = 50
        self.screen = None
        pygame.font.init()
        self.font = pygame.font.SysFont('Arial', 16)
        self.title_font = pygame.font.SysFont('Arial', 24, bold=True)
        self.info_font = pygame.font.SysFont('Arial', 16)
        self.small_font = pygame.font.SysFont('Arial', 12)

        #define colors

        self.colors = {
            'background': (16, 24, 32),       # Deep navy blue
            'grid': (26, 36, 46),             # Slightly lighter navy for grid
            'agent': (102, 255, 178),         # Bright mint green for agent
            'target': (255, 89, 94),          # Coral red for target
            'obstacle': (184, 115, 51),       # Warm orange for obstacle dots
            'obstacle_zone': (184, 115, 51, 50), # Warm brown with more opacity for zones
            'text': (220, 230, 240),          # Off-white for text
            'warning': (255, 89, 94),         # Coral red for warnings
            'trajectory': (103, 140, 255, 160) # Brighter blue for trajectory
        }
        '''
        self.colors = {
            'background': (15, 15, 35),      # Dark blue-gray
            'grid': (30, 30, 50),            # Lighter grid lines
            'agent': (0, 255, 0),            # Green for agent
            'target': (255, 100, 100),       # Soft red for target
            'obstacle': (255, 165, 0),       # Orange for obstacles
            'obstacle_zone': (255, 165, 0, 40),  # Semi-transparent dark red
            'text': (200, 200, 200),         # Light gray for text
            'warning': (255, 60, 60),        # Red for warnings
            'trajectory': (100, 100, 255, 128)  # Semi-transparent blue
        }
        '''

        self.trajectory = []
        self.sprite_original = None
        self.sprite_size = (20, 25)

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
        self._agent_state = {
            "position": np.array([-1, -1], dtype=float),
            "heading": 0
        }
        self._target_state = {
            "position": np.array(self.rng.uniform(0, self.size, size=2))
        }
        #self.observation_space[-1][-1] = -1 #means static
        '''
        Action Space:
        Stay, Slight Left, Slight Right, Hard Left, Hard Right.
        
        t.b.d: slight turns are 10 degrees, while hard turns are 30 degrees.
        '''
        self.action_space = gym.spaces.Discrete(5)
        self._obstacles = {
            "static":[np.array(self.rng.uniform(0, self.size, size=2)) for _ in range(self.num_stat_obs)],
            "motional": None
        }
        
    def _get_obs(self):
        pos, heading = self._agent_state["position"], self._agent_state["heading"]
        target = self._target_state
        obs = np.zeros(self.observation_space.shape, dtype=self.observation_space.dtype)

        for i, x in enumerate(self._obstacles["static"]):
            obs[i][0] = np.linalg.norm(pos - x)
            obs[i][1] = calculate_heading(pos, x)
            obs[i][2] = -1
            assert i < obs.shape[0] - 1

        #for i, x in enumerate(self._obstacles["motional"]):
        if self._obstacles["motional"] is not None:
            raise NotImplementedError("Motional obstacles not implemented, but observing.")

        obs[-1][0] = np.linalg.norm(pos - target["position"])
        obs[-1][1] = calculate_heading(pos, target["position"])
        obs[-1][2] = -1

        #self.observation_space = obs
        #return self.observation_space
        assert obs.shape == (self.num_mot_obs + self.num_stat_obs + 1, 3)
        return obs
    
    def get_position_debug(self):
        return self._agent_state["position"], self._target_state["position"], self._agent_state["heading"], calculate_heading(self._agent_state["position"], self._target_state["position"])
    
    def _get_info(self):
        '''
        return {
            "total_reward": self.episodic_reward,
            "total_steps": self.episodic_step,
            "agent_state": self._agent_state,
            "target_state": self._target_state
        }
        '''
        return self.episodic_info
    
    def reset(self, seed: Optional[int] = None, motional_obstacles = False):
        super().reset(seed=seed)
        self.episodic_step = 0
        self.episodic_reward = 0
        self._target_state = {
            "position": np.array(self.rng.uniform(0, self.size, size=2)),
        }
        self._agent_state = {
            "position": np.array(self.rng.uniform(0, self.size, size=2))
        }
        #self._target_state = np.array(self.rng.uniform(0, self.size, size=2))
        self._agent_state["heading"] = calculate_heading(self._agent_state["position"], self._target_state["position"])
        
        #initialize obstacles related information
        if not motional_obstacles:
            self._obstacles = {
                "static":[np.array(self.rng.uniform(0, self.size, size=2)) for _ in range(self.num_stat_obs)],
                "motional": None
            }
        else:
            raise NotImplementedError("Motional obstacles not implemented, but initializing.")
        _, __, heading, target = self.get_position_debug()
        assert heading == target
        self.trajectory = []
        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def step(self, action, alpha = 0.5, beta = 1.0, gamma = 4.0):
        '''
        Action Space:
        Stay, Slight Left, Slight Right, Hard Left, Hard Right.
        
        t.b.d: slight turns are 10 degrees, while hard turns are 30 degrees.
        '''
        pos, heading = self._agent_state["position"], self._agent_state["heading"]
        angle_change = [0, self.slight_turn, -self.slight_turn,
                        self.hard_turn, -self.hard_turn]
        #print(action)
        _new_state = {
            "position": calculate_position(pos, heading, self.speed),
            "heading": heading + angle_change[action]
        }
        self._agent_state = _new_state
        if self._obstacles["motional"] is not None:
            self._obstacles["static"].extend(self._obstacles["motional"])
        lose = any(arrive(_new_state["position"], x, self.tolerance)
                for x in self._obstacles["static"])
        win = arrive(_new_state["position"],
                           self._target_state["position"], self.tolerance)
        terminated = win or lose
        truncated = self.episodic_step > self.max_steps

        #check boundaries
        if self._check_boundaries():
            boundary_viol = -100.0
            terminated = True
        else:
            boundary_viol = 0

        obs = self._get_obs()
        target_distance = obs[-1][0]
        target_heading = obs[-1][1]
        reward_target = alpha * (self.size - target_distance) + beta * (360 - target_heading)
        penalty_obstacle = 0
        for i, x in enumerate(obs):
            if i == obs.shape[0] - 1:
                break
            dis = x[0] if x[0] < self.radius else 0
            penalty_obstacle += self.radius * self.radius - dis * dis
        penalty_obstacle *= gamma 
        reward_terminate = 500 if win else(-500 if lose else 0)

        reward = reward_target + penalty_obstacle + reward_terminate + boundary_viol
        self.episodic_step += 1
        self.episodic_reward += reward
        if terminated or truncated:
            self.episodic_info["total_reward"] = self.episodic_reward
            self.episodic_info["total_steps"] = self.episodic_step
        return obs, reward, terminated, truncated, {}

    def _check_boundaries(self):
        pos = self._agent_state["position"]
        return (pos[0] <= 0 or pos[0] >= self.size - 1 or
                pos[1] <= 0 or pos[1] >= self.size - 1)

    def render(self):
        if self.screen is None:
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            pygame.display.set_caption('Discrete Approach Environment')

        # Clear screen
        self.screen.fill(self.colors['background'])

        # Draw grid
        self._draw_grid()

        # Draw boundaries and info panels
        self._draw_boundaries()
        self._draw_info_panels()

        # Draw obstacles
        self._draw_obstacles()

        # Draw agent and target
        agent_pos = self._world_to_screen(self._agent_state["position"])
        self.trajectory.append(agent_pos)
        #self._draw_agent(agent_pos, self._agent_state["heading"])
        self._draw_aircraft(agent_pos, self._agent_state["heading"])

        target_pos = self._world_to_screen(self._target_state["position"])
        self._draw_target(target_pos)

        # Draw trajectory
        self._draw_trajectory()

        pygame.display.flip()

    def _world_to_screen(self, pos):
        """Convert world coordinates to screen coordinates"""
        x = self.padding + (pos[0] / self.size) * (self.screen_width - 2 * self.padding)
        y = self.screen_height - (self.padding + (pos[1] / self.size) * (self.screen_height - 2 * self.padding))
        return (int(x), int(y))

    def _draw_grid(self):
        # Draw coordinate grid with appropriate spacing for discrete environment
        cell_size = (self.screen_width - 2 * self.padding) / self.size

        # Draw vertical lines
        for x in range(self.size + 1):
            screen_x = self.padding + x * cell_size
            pygame.draw.line(self.screen, self.colors['grid'],
                             (screen_x, self.padding),
                             (screen_x, self.screen_height - self.padding))

        # Draw horizontal lines
        for y in range(self.size + 1):
            screen_y = self.padding + y * cell_size
            pygame.draw.line(self.screen, self.colors['grid'],
                             (self.padding, screen_y),
                             (self.screen_width - self.padding, screen_y))

    def _draw_boundaries(self):
        """Draw the environment boundaries"""
        pygame.draw.rect(self.screen, self.colors['grid'],
                         (self.padding, self.padding,
                          self.screen_width - 2*self.padding,
                          self.screen_height - 2*self.padding), 2)

    def _draw_obstacles(self):
        """Draw all obstacles with simple influence zones"""
        # Create a surface for the semi-transparent influence zones
        influence_surface = pygame.Surface((self.screen_width, self.screen_height), pygame.SRCALPHA)

        for obs_pos in self._obstacles["static"]:
            screen_pos = self._world_to_screen(obs_pos)

            # Draw influence zone as a semi-transparent circle
            influence_radius = int(self.radius * (self.screen_width - 2*self.padding) / self.size)
            pygame.draw.circle(influence_surface, self.colors['obstacle_zone'],
                               screen_pos, influence_radius)

            # Draw obstacle as a small solid dot
            pygame.draw.circle(self.screen, self.colors['obstacle'], screen_pos, 8)

        # Blit the influence surface onto the main screen
        self.screen.blit(influence_surface, (0, 0))

    def _draw_agent(self, pos, heading):
        """Draw the agent with heading indicator"""
        # Draw agent
        pygame.draw.circle(self.screen, self.colors['agent'], pos, 8)

        # Draw heading indicator
        heading_rad = math.radians(heading)
        end_pos = (pos[0] + 15 * math.cos(heading_rad),
                   pos[1] - 15 * math.sin(heading_rad))
        pygame.draw.line(self.screen, self.colors['agent'], pos, end_pos, 2)

    def _draw_target(self, pos):
        """Draw the target"""
        size = 12
        pygame.draw.circle(self.screen, self.colors['target'], pos, size, 2)
        pygame.draw.circle(self.screen, self.colors['target'], pos, 3)

    def _draw_trajectory(self):
        """Draw the agent's trajectory"""
        if len(self.trajectory) > 1:
            pygame.draw.lines(self.screen, self.colors['trajectory'], False, self.trajectory, 2)

    def _draw_info_panels(self):
        """Draw information panels with state and navigation info"""
        # Create surfaces with alpha channel
        state_panel = pygame.Surface((200, 120), pygame.SRCALPHA)
        nav_panel = pygame.Surface((200, 100), pygame.SRCALPHA)

        # Fill with semi-transparent dark background
        panel_bg_color = (26, 36, 46, 200)
        pygame.draw.rect(state_panel, panel_bg_color, (0, 0, 250, 150))
        pygame.draw.rect(nav_panel, panel_bg_color, (0, 0, 250, 150))

        # State Panel
        title = self.title_font.render("State", True, self.colors['text'])
        state_panel.blit(title, (10, 5))

        # Convert numpy values to float before formatting
        pos_x = float(self._agent_state['position'][0])
        pos_y = float(self._agent_state['position'][1])
        heading = float(self._agent_state['heading'][0] if isinstance(self._agent_state['heading'], np.ndarray)
                        else self._agent_state['heading'])
        ep_reward = float(self.episodic_reward)

        # State information
        info_texts = [
            f"Position: ({pos_x:.1f}, {pos_y:.1f})",
            f"Heading: {heading:.1f}°",
            #f"Step: {self.episodic_step}/{self.max_steps}",
            f"Reward: {ep_reward:.1f}"
        ]

        # Render state information
        for i, text in enumerate(info_texts):
            text_surface = self.info_font.render(text, True, self.colors['text'])
            state_panel.blit(text_surface, (10, 35 + i * 25))

        # Navigation Panel
        nav_title = self.title_font.render("Navigation", True, self.colors['text'])
        nav_panel.blit(nav_title, (10, 5))

        # Calculate distance
        distance = float(np.linalg.norm(self._agent_state['position'] - self._target_state['position']))

        nav_texts = [
            f"Distance: {distance:.1f}",
            f"Total Steps: {self.episodic_step}"
        ]

        # Render navigation information
        for i, text in enumerate(nav_texts):
            text_surface = self.info_font.render(text, True, self.colors['text'])
            nav_panel.blit(text_surface, (10, 35 + i * 25))

        # Add borders
        border_color = (40, 50, 60, 255)
        pygame.draw.rect(state_panel, border_color, (0, 0, 250, 150), 1)
        pygame.draw.rect(nav_panel, border_color, (0, 0, 250, 150), 1)

        # Blit panels to screen
        self.screen.blit(state_panel, (10, 10))
        self.screen.blit(nav_panel, (10, 170))


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

    def _draw_aircraft(self, pos, heading, color=(0, 255, 0)):
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
    
    def need_rl(self):
        obs = self._get_obs()
        ret = False
        for i, x in enumerate(obs):
            if i == obs.shape[0] - 1:
                break
            if x[0] < self.radius :
                ret = True
                break
        return ret

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
            self.screen = None
        self.trajectory = []