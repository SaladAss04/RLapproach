import pygame
import numpy as np
from .themes import ColorScheme

class ATCRenderer:  # Changed from ModernRenderer to ATCRenderer
    """Modern rendering system for ATC environment using Pygame."""
    
    def __init__(self, screen_width=800, screen_height=600):
        """Initialize the renderer."""
        pygame.init()
        pygame.font.init()
        
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.screen = None
        self.font = pygame.font.SysFont('Arial', 14)
        self.padding = 40
        
    def setup(self, world_x_min, world_x_max, world_y_min, world_y_max):
        """Setup display based on world coordinates."""
        world_width = world_x_max - world_x_min
        world_height = world_y_max - world_y_min
        
        # Calculate scaling to maintain aspect ratio
        self.scale = min(
            (self.screen_width - 2 * self.padding) / world_width,
            (self.screen_height - 2 * self.padding) / world_height
        )
        
        # Initialize display if not already done
        if self.screen is None:
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            pygame.display.set_caption('ATC Environment')
        
        # Store world boundaries for coordinate conversion
        self.world_x_min = world_x_min
        self.world_y_min = world_y_min
    
    def render(self, env_state, mode='human'):
        """Main render method - entry point from environment."""
        if not hasattr(self, 'scale'):
            self.setup(0, env_state['config']['airspace_size'],
                      0, env_state['config']['airspace_size'])
        
        # Clear screen
        self.render_frame()
        
        # Render environment elements
        self.render_runway(
            env_state['config']['faf_position'][0] - 5,  # 5nm before FAF
            env_state['config']['faf_position'][1],
            env_state['config']['runway_heading']
        )
        self.render_faf(*env_state['config']['faf_position'])
        
        # Render aircraft if it exists
        if env_state['aircraft']:
            self.render_airplane(
                env_state['aircraft'].x,
                env_state['aircraft'].y,
                env_state['aircraft'].heading,
                env_state['aircraft'].altitude,
                env_state['aircraft'].speed,
                env_state['aircraft'].position_history
            )
        
        # Render information overlay
        self.render_info({
            'Step': env_state['steps'],
            'Reward': f"{env_state.get('reward', 0):.1f}"
        })
        
        return self.finish_render(mode)

class ModernRenderer:
    """Modern rendering system for ATC environment using Pygame."""
    
    def __init__(self, screen_width=800, screen_height=600):
        """Initialize the renderer."""
        pygame.init()
        pygame.font.init()
        
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.screen = None
        self.font = pygame.font.SysFont('Arial', 14)
        self.padding = 40
        
    def setup(self, world_x_min, world_x_max, world_y_min, world_y_max):
        """Setup display based on world coordinates."""
        world_width = world_x_max - world_x_min
        world_height = world_y_max - world_y_min
        
        # Calculate scaling to maintain aspect ratio
        self.scale = min(
            (self.screen_width - 2 * self.padding) / world_width,
            (self.screen_height - 2 * self.padding) / world_height
        )
        
        # Initialize display if not already done
        if self.screen is None:
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            pygame.display.set_caption('ATC Environment')
        
        # Store world boundaries for coordinate conversion
        self.world_x_min = world_x_min
        self.world_y_min = world_y_min
        
    def world_to_screen(self, x, y):
        """Convert world coordinates to screen coordinates."""
        screen_x = (x - self.world_x_min) * self.scale + self.padding
        # Flip y-coordinate because pygame's origin is top-left
        screen_y = self.screen_height - ((y - self.world_y_min) * self.scale + self.padding)
        return int(screen_x), int(screen_y)

    def render_airplane(self, x, y, heading, altitude, speed, trail=None):
        """Render an airplane with its trail and information."""
        # Convert position to screen coordinates
        screen_x, screen_y = self.world_to_screen(x, y)
        
        # Draw trail if available
        if trail:
            points = [self.world_to_screen(px, py) for px, py in trail]
            if len(points) > 1:
                pygame.draw.lines(self.screen, ColorScheme.airplane, False, points, 1)
        
        # Draw airplane triangle
        size = 10
        heading_rad = np.radians(-heading + 90)  # Convert to pygame angle convention
        points = [
            (screen_x + size * np.cos(heading_rad),
             screen_y + size * np.sin(heading_rad)),
            (screen_x + size * np.cos(heading_rad + 2.6),
             screen_y + size * np.sin(heading_rad + 2.6)),
            (screen_x + size * np.cos(heading_rad - 2.6),
             screen_y + size * np.sin(heading_rad - 2.6))
        ]
        pygame.draw.polygon(self.screen, ColorScheme.airplane, points)
        
        # Draw information label
        self.render_text(
            f"{int(altitude/100)},{int(speed)}", 
            screen_x + 15, 
            screen_y - 10
        )

    def render_runway(self, x, y, heading, length=3.0):
        """Render runway and extended centerline."""
        runway_x, runway_y = self.world_to_screen(x, y)
        runway_length = int(length * self.scale)
        runway_width = 6
        
        # Draw runway
        heading_rad = np.radians(-heading + 90)
        dx = runway_length * np.cos(heading_rad)
        dy = runway_length * np.sin(heading_rad)
        
        pygame.draw.line(
            self.screen,
            ColorScheme.runway,
            (runway_x - dx/2, runway_y - dy/2),
            (runway_x + dx/2, runway_y + dy/2),
            runway_width
        )
        
        # Draw extended centerline (dashed)
        centerline_length = runway_length * 3
        dash_length = 20
        gap_length = 10
        current_dist = runway_length/2
        
        while current_dist < centerline_length:
            start_x = runway_x + (current_dist * np.cos(heading_rad))
            start_y = runway_y + (current_dist * np.sin(heading_rad))
            end_x = start_x + (dash_length * np.cos(heading_rad))
            end_y = start_y + (dash_length * np.sin(heading_rad))
            
            pygame.draw.line(
                self.screen,
                ColorScheme.runway,
                (int(start_x), int(start_y)),
                (int(end_x), int(end_y)),
                2
            )
            current_dist += dash_length + gap_length

    def render_faf(self, x, y):
        """Render Final Approach Fix marker."""
        screen_x, screen_y = self.world_to_screen(x, y)
        size = 10
        
        # Draw FAF as a cross
        pygame.draw.line(
            self.screen,
            ColorScheme.runway,
            (screen_x - size, screen_y),
            (screen_x + size, screen_y),
            2
        )
        pygame.draw.line(
            self.screen,
            ColorScheme.runway,
            (screen_x, screen_y - size),
            (screen_x, screen_y + size),
            2
        )

    def render_text(self, text, x, y, color=ColorScheme.label):
        """Render text at specified position."""
        text_surface = self.font.render(text, True, color)
        self.screen.blit(text_surface, (x, y))

    def render_info(self, info_dict):
        """Render information overlay."""
        y = 10
        for key, value in info_dict.items():
            text = f"{key}: {value}"
            self.render_text(text, 10, y)
            y += 20

    def render_frame(self):
        """Clear screen and prepare for rendering."""
        self.screen.fill(ColorScheme.background_inactive)

    def finish_render(self, mode='human'):
        """Complete the rendering and return appropriate output."""
        if mode == 'human':
            pygame.display.flip()
            return None
        elif mode == 'rgb_array':
            return np.transpose(
                pygame.surfarray.array3d(self.screen),
                axes=(1, 0, 2)
            )

    def close(self):
        """Clean up pygame resources."""
        pygame.quit()