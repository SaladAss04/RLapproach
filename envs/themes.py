class ColorScheme:
    """Color definitions for the rendering system."""
    # Convert from 0-1 range to 0-255 range for pygame
    background_inactive = (26, 26, 26)     # Dark gray
    background_active = (38, 38, 38)       # Slightly lighter gray
    lines_info = (204, 204, 204)          # Light gray
    mva = (102, 102, 102)                 # Medium gray
    runway = (230, 230, 230)              # Almost white
    airplane = (0, 204, 0)                # Green
    label = (255, 255, 255)               # White