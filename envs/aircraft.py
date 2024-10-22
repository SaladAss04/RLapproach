import numpy as np
from dataclasses import dataclass
from typing import Tuple, List

@dataclass
class AircraftConfig:
    """Aircraft performance configuration parameters"""
    min_speed: float = 100.0  # Minimum speed in knots
    max_speed: float = 300.0  # Maximum speed in knots
    min_altitude: float = 0.0  # Minimum altitude in feet
    max_altitude: float = 38000.0  # Maximum altitude in feet
    max_turn_rate: float = 3.0  # Maximum turn rate in degrees/second
    max_climb_rate: float = 15.0  # Maximum climb rate in feet/second
    max_descent_rate: float = -41.0  # Maximum descent rate in feet/second
    max_acceleration: float = 5.0  # Maximum acceleration in knots/second
    max_deceleration: float = -5.0  # Maximum deceleration in knots/second

class Aircraft:
    """
    Aircraft model with basic physics and constraints.
    
    Handles aircraft movement, speed changes, altitude changes, and turning
    while respecting aircraft performance limitations.
    """
    
    def __init__(self, 
                 initial_position: Tuple[float, float],  # (x, y) in nautical miles
                 initial_altitude: float,  # feet
                 initial_heading: float,  # degrees
                 initial_speed: float,  # knots
                 config: AircraftConfig = None,
                 timestep: float = 1.0):  # seconds
        """Initialize aircraft with position and configuration."""
        self.config = config or AircraftConfig()
        self.timestep = timestep
        
        # State variables
        self.x, self.y = initial_position
        self.altitude = initial_altitude
        self.heading = initial_heading
        self.speed = initial_speed
        
        # Validation
        self._validate_state()
        
        # Track aircraft path for visualization
        self.position_history: List[Tuple[float, float]] = []
    
    def _validate_state(self) -> None:
        """Validate aircraft state against configuration limits."""
        if not (self.config.min_speed <= self.speed <= self.config.max_speed):
            raise ValueError(f"Speed {self.speed} outside limits [{self.config.min_speed}, {self.config.max_speed}]")
        
        if not (self.config.min_altitude <= self.altitude <= self.config.max_altitude):
            raise ValueError(f"Altitude {self.altitude} outside limits [{self.config.min_altitude}, {self.config.max_altitude}]")
    
    def update_speed(self, target_speed: float) -> None:
        """
        Update aircraft speed respecting acceleration limits.
        
        Args:
            target_speed: Desired speed in knots
        """
        if not (self.config.min_speed <= target_speed <= self.config.max_speed):
            raise ValueError(f"Target speed {target_speed} outside limits")
        
        # Calculate maximum allowed speed change this timestep
        max_change = self.config.max_acceleration * self.timestep
        min_change = self.config.max_deceleration * self.timestep
        
        # Apply speed change with limits
        speed_change = np.clip(target_speed - self.speed, min_change, max_change)
        self.speed = self.speed + speed_change
    
    def update_altitude(self, target_altitude: float) -> None:
        """
        Update aircraft altitude respecting climb/descent rates.
        
        Args:
            target_altitude: Desired altitude in feet
        """
        if not (self.config.min_altitude <= target_altitude <= self.config.max_altitude):
            raise ValueError(f"Target altitude {target_altitude} outside limits")
        
        # Calculate maximum allowed altitude change this timestep
        max_climb = self.config.max_climb_rate * self.timestep
        max_descent = self.config.max_descent_rate * self.timestep
        
        # Apply altitude change with limits
        altitude_change = np.clip(target_altitude - self.altitude, max_descent, max_climb)
        self.altitude = self.altitude + altitude_change
    
    def update_heading(self, target_heading: float) -> None:
        """
        Update aircraft heading respecting turn rate limits.
        
        Args:
            target_heading: Desired heading in degrees (0-360)
        """
        # Normalize target heading to 0-360 range
        target_heading = target_heading % 360
        
        # Calculate required heading change considering wraparound
        heading_change = ((target_heading - self.heading + 180) % 360) - 180
        
        # Apply turn rate limits
        max_change = self.config.max_turn_rate * self.timestep
        heading_change = np.clip(heading_change, -max_change, max_change)
        
        # Update heading and normalize to 0-360
        self.heading = (self.heading + heading_change) % 360
    
    def step(self) -> None:
        """Update aircraft position based on current speed and heading."""
        # Store current position for visualization
        self.position_history.append((self.x, self.y))
        
        # Convert speed from knots to nm/second and calculate movement
        speed_nm_sec = (self.speed / 3600.0) * self.timestep
        
        # Calculate position change using heading
        heading_rad = np.radians(self.heading)
        self.x += speed_nm_sec * np.cos(heading_rad)
        self.y += speed_nm_sec * np.sin(heading_rad)
    
    @property
    def position(self) -> Tuple[float, float]:
        """Current aircraft position."""
        return (self.x, self.y)
    
    @property
    def state(self) -> Tuple[float, float, float, float, float]:
        """Complete aircraft state: (x, y, altitude, heading, speed)."""
        return (self.x, self.y, self.altitude, self.heading, self.speed)