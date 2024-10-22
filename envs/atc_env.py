import gym
import numpy as np
from gym import spaces
from typing import Dict, Tuple, Optional
from .aircraft import Aircraft, AircraftConfig

class ATCEnv(gym.Env):
    """
    Air Traffic Control environment for reinforcement learning.
    
    The task is to guide aircraft to the final approach fix (FAF) at the correct
    altitude and angle for landing.
    """
    
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }
    
    def __init__(self):
        super(ATCEnv, self).__init__()
        
        # Environment configuration
        self.config = {
            'airspace_size': 50.0,  # Size of square airspace in nautical miles
            'min_altitude': 0,      # Minimum altitude in feet
            'max_altitude': 20000,  # Maximum altitude in feet
            'faf_position': (40, 25),  # Final Approach Fix position
            'runway_heading': 270,   # Runway heading in degrees
            'min_separation': 3.0,   # Minimum separation in nautical miles
            'max_steps': 500        # Maximum steps per episode
        }
        
        # Define action space: (heading change, altitude change, speed change)
        self.action_space = spaces.Box(
            low=np.array([-30, -2000, -20]),  # heading °, altitude ft, speed kt
            high=np.array([30, 2000, 20]),
            dtype=np.float32
        )
        
        # Define observation space
        # [x, y, altitude, heading, speed, distance_to_faf, angle_to_faf, runway_alignment]
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 100, 0, -180, -180]),
            high=np.array([
                self.config['airspace_size'],
                self.config['airspace_size'],
                self.config['max_altitude'],
                360,
                300,
                np.sqrt(2) * self.config['airspace_size'],
                180,
                180
            ]),
            dtype=np.float32
        )
        
        # Initialize render system
        self.viewer = None
        
        # Initialize state
        self.aircraft: Optional[Aircraft] = None
        self.steps = 0
        self.reset()
    
    def reset(self):
        """Reset environment to initial state."""
        # Create new aircraft in random initial position
        initial_pos = self._generate_initial_position()
        self.aircraft = Aircraft(
            initial_position=initial_pos,
            initial_altitude=np.random.uniform(5000, 10000),
            initial_heading=np.random.uniform(0, 360),
            initial_speed=250
        )
        
        self.steps = 0
        return self._get_observation()
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute one step in the environment."""
        self.steps += 1
        
        # Apply actions
        # Convert relative actions to absolute targets
        target_heading = (self.aircraft.heading + action[0]) % 360
        target_altitude = self.aircraft.altitude + action[1]
        target_speed = self.aircraft.speed + action[2]
        
        # Update aircraft state
        self.aircraft.update_heading(target_heading)
        self.aircraft.update_altitude(target_altitude)
        self.aircraft.update_speed(target_speed)
        self.aircraft.step()
        
        # Get new state
        observation = self._get_observation()
        
        # Calculate reward
        reward = self._calculate_reward()
        
        # Check if episode is done
        done = self._is_done()
        
        return observation, reward, done, {}
    
    def _get_observation(self) -> np.ndarray:
        """Convert current state to observation vector."""
        x, y = self.aircraft.position
        dist_to_faf = self._distance_to_faf()
        angle_to_faf = self._angle_to_faf()
        runway_alignment = self._runway_alignment()
        
        return np.array([
            x,
            y,
            self.aircraft.altitude,
            self.aircraft.heading,
            self.aircraft.speed,
            dist_to_faf,
            angle_to_faf,
            runway_alignment
        ], dtype=np.float32)
    
    def _calculate_reward(self) -> float:
        """Calculate reward based on current state."""
        reward = 0.0
        
        # Distance-based reward
        dist_to_faf = self._distance_to_faf()
        reward -= 0.1 * dist_to_faf  # Small penalty for distance
        
        # Alignment reward
        runway_alignment = abs(self._runway_alignment())
        if runway_alignment < 45:  # Only reward good alignment
            reward += (45 - runway_alignment) / 45.0
        
        # Altitude reward
        target_alt = self._ideal_approach_altitude()
        alt_diff = abs(self.aircraft.altitude - target_alt)
        reward -= 0.001 * alt_diff
        
        # Success reward
        if self._is_at_faf():
            reward += 100.0
        
        return reward
    
    def _is_done(self) -> bool:
        """Check if episode should end."""
        # Episode ends if:
        # 1. Aircraft reaches FAF
        # 2. Aircraft leaves airspace
        # 3. Max steps reached
        return (self._is_at_faf() or
                self._is_out_of_bounds() or
                self.steps >= self.config['max_steps'])
    
    def _generate_initial_position(self) -> Tuple[float, float]:
        """Generate random initial position for aircraft."""
        # Generate position on edge of airspace
        if np.random.random() < 0.5:
            # Random position on vertical edges
            x = np.random.choice([0, self.config['airspace_size']])
            y = np.random.uniform(0, self.config['airspace_size'])
        else:
            # Random position on horizontal edges
            x = np.random.uniform(0, self.config['airspace_size'])
            y = np.random.choice([0, self.config['airspace_size']])
        return (x, y)
    
    def _distance_to_faf(self) -> float:
        """Calculate distance to Final Approach Fix."""
        x, y = self.aircraft.position
        faf_x, faf_y = self.config['faf_position']
        return np.hypot(x - faf_x, y - faf_y)
    
    def _angle_to_faf(self) -> float:
        """Calculate relative angle to Final Approach Fix."""
        x, y = self.aircraft.position
        faf_x, faf_y = self.config['faf_position']
        angle = np.degrees(np.arctan2(faf_y - y, faf_x - x))
        return (angle - self.aircraft.heading + 180) % 360 - 180
    
    def _runway_alignment(self) -> float:
        """Calculate alignment with runway heading."""
        return (self.aircraft.heading - self.config['runway_heading'] + 180) % 360 - 180
    
    def _ideal_approach_altitude(self) -> float:
        """Calculate ideal altitude at current position."""
        dist = self._distance_to_faf()
        # 3-degree glide slope
        return 3000 + dist * 318  # 318 ft/nm is approximately 3 degrees
    
    def _is_at_faf(self) -> bool:
        """Check if aircraft is at the Final Approach Fix."""
        dist = self._distance_to_faf()
        angle = abs(self._runway_alignment())
        alt_diff = abs(self.aircraft.altitude - 3000)  # FAF altitude
        
        return (dist < 0.5 and  # Within 0.5 nm
                angle < 10 and  # Within 10 degrees
                alt_diff < 200)  # Within 200 feet
    
    def _is_out_of_bounds(self) -> bool:
        """Check if aircraft has left the airspace."""
        x, y = self.aircraft.position
        return (x < 0 or x > self.config['airspace_size'] or
                y < 0 or y > self.config['airspace_size'])