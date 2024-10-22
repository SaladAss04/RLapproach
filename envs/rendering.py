import numpy as np
from gym.envs.classic_control import rendering
from typing import Tuple, Optional

class ATCRenderer:
    """Renderer for the ATC environment using OpenGL/Pyglet."""
    
    def __init__(self, screen_width: int = 600, padding: int = 40):
        self.screen_width = screen_width
        self.padding = padding
        self.viewer: Optional[rendering.Viewer] = None
        
        # Colors
        self.colors = {
            'background': (0.1, 0.1, 0.1),  # Dark background
            'aircraft': (0.0, 0.8, 0.0),    # Green for aircraft
            'runway': (0.8, 0.8, 0.8),      # Light grey for runway
            'faf': (0.8, 0.0, 0.0),         # Red for FAF
            'text': (1.0, 1.0, 1.0),        # White for text
        }
    
    def render(self, 
               env_state: dict,
               mode: str = 'human') -> Optional[np.ndarray]:
        """
        Render the environment state.
        
        Args:
            env_state: Dictionary containing environment state
            mode: 'human' for window display, 'rgb_array' for array return
            
        Returns:
            numpy array of rendered frame if mode is 'rgb_array', else None
        """
        if self.viewer is None:
            self._setup_viewer(env_state['config'])
        
        # Clear the viewer
        self.viewer.geoms = []
        
        # Draw environment elements
        self._draw_runway(env_state['config'])
        self._draw_faf(env_state['config'])
        self._draw_aircraft(env_state['aircraft'])
        self._draw_info(env_state)
        
        return self.viewer.render(return_rgb_array=(mode == 'rgb_array'))
    
    def _setup_viewer(self, config: dict):
        """Initialize the viewing window."""
        world_width = config['airspace_size']
        world_height = config['airspace_size']
        
        # Calculate scaling to fit screen
        self.scale = (self.screen_width - 2 * self.padding) / world_width
        screen_height = int(world_height * self.scale) + 2 * self.padding
        
        self.viewer = rendering.Viewer(self.screen_width, screen_height)
        
        # Set background color
        background = rendering.FilledPolygon([
            (0, 0),
            (0, screen_height),
            (self.screen_width, screen_height),
            (self.screen_width, 0)
        ])
        background.set_color(*self.colors['background'])
        self.viewer.add_geom(background)
    
    def _world_to_screen(self, x: float, y: float) -> Tuple[float, float]:
        """Convert world coordinates to screen coordinates."""
        screen_x = x * self.scale + self.padding
        screen_y = y * self.scale + self.padding
        return (screen_x, screen_y)
    
    def _draw_aircraft(self, aircraft):
        """Draw aircraft symbol and information."""
        x, y = self._world_to_screen(aircraft.x, aircraft.y)
        
        # Draw aircraft symbol (triangle)
        size = 10
        heading_rad = np.radians(aircraft.heading)
        points = [
            (x + size * np.cos(heading_rad),
             y + size * np.sin(heading_rad)),
            (x + size * np.cos(heading_rad + 2.6),
             y + size * np.sin(heading_rad + 2.6)),
            (x + size * np.cos(heading_rad - 2.6),
             y + size * np.sin(heading_rad - 2.6))
        ]
        aircraft_shape = rendering.FilledPolygon(points)
        aircraft_shape.set_color(*self.colors['aircraft'])
        self.viewer.add_geom(aircraft_shape)
        
        # Draw altitude and speed info
        label = rendering.Label(
            f"{int(aircraft.altitude/100)},{int(aircraft.speed)}", 
            x=x + 15, 
            y=y, 
            color=self.colors['text']
        )
        self.viewer.add_geom(label)
        
        # Draw trail
        if len(aircraft.position_history) > 1:
            trail_points = []
            for pos in aircraft.position_history[-20:]:  # Last 20 positions
                trail_x, trail_y = self._world_to_screen(pos[0], pos[1])
                trail_points.append((trail_x, trail_y))
            trail = rendering.PolyLine(trail_points, False)
            trail.set_color(*self.colors['aircraft'])
            trail.set_linewidth(1)
            self.viewer.add_geom(trail)
    
    def _draw_runway(self, config: dict):
        """Draw runway and extended centerline."""
        runway_x, runway_y = self._world_to_screen(
            config['faf_position'][0] - 5,  # 5nm before FAF
            config['faf_position'][1]
        )
        runway_length = 20 * self.scale  # 2nm long
        runway_width = 4
        
        # Convert runway heading to radians
        heading_rad = np.radians(config['runway_heading'])
        
        # Draw runway
        runway = rendering.FilledPolygon([
            (runway_x - runway_length/2, runway_y - runway_width/2),
            (runway_x - runway_length/2, runway_y + runway_width/2),
            (runway_x + runway_length/2, runway_y + runway_width/2),
            (runway_x + runway_length/2, runway_y - runway_width/2)
        ])
        runway.set_color(*self.colors['runway'])
        self.viewer.add_geom(runway)
        
        # Draw extended centerline
        centerline_points = []
        for d in range(0, 150, 10):  # Dashed line
            start_x = runway_x + (d * self.scale) * np.cos(heading_rad)
            start_y = runway_y + (d * self.scale) * np.sin(heading_rad)
            end_x = start_x + (5 * self.scale) * np.cos(heading_rad)
            end_y = start_y + (5 * self.scale) * np.sin(heading_rad)
            line = rendering.PolyLine([(start_x, start_y), (end_x, end_y)], False)
            line.set_color(*self.colors['runway'])
            self.viewer.add_geom(line)
    
    def _draw_faf(self, config: dict):
        """Draw Final Approach Fix marker."""
        faf_x, faf_y = self._world_to_screen(*config['faf_position'])
        
        # Draw FAF symbol (cross)
        size = 8
        for angle in [0, np.pi/2]:  # Two perpendicular lines
            start_x = faf_x + size * np.cos(angle)
            start_y = faf_y + size * np.sin(angle)
            end_x = faf_x - size * np.cos(angle)
            end_y = faf_y - size * np.sin(angle)
            line = rendering.PolyLine([(start_x, start_y), (end_x, end_y)], False)
            line.set_color(*self.colors['faf'])
            line.set_linewidth(2)
            self.viewer.add_geom(line)
    
    def _draw_info(self, env_state: dict):
        """Draw environment information."""
        # Draw step count and reward
        step_label = rendering.Label(
            f"Step: {env_state['steps']}", 
            x=10, 
            y=self.screen_width - 20,
            color=self.colors['text']
        )
        self.viewer.add_geom(step_label)
        
        if 'reward' in env_state:
            reward_label = rendering.Label(
                f"Reward: {env_state['reward']:.1f}", 
                x=10, 
                y=self.screen_width - 40,
                color=self.colors['text']
            )
            self.viewer.add_geom(reward_label)
    
    def close(self):
        """Clean up viewer resources."""
        if self.viewer:
            self.viewer.close()
            self.viewer = None