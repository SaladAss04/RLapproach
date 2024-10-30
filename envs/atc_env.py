import gym
import numpy as np
from gym import spaces
from typing import Dict, Tuple, Optional
from .aircraft import Aircraft, AircraftConfig

class ATCEnv(gym.Env):  # Make sure the class name is ATCEnv
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
            'max_steps': 500,       # Maximum steps per episode
            # Add missing coordinates for renderer
            'airspace_x_min': 0,
            'airspace_x_max': 50.0,
            'airspace_y_min': 0,
            'airspace_y_max': 50.0
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
        
        # Initialize state variables
        self.viewer = None
        self.aircraft: Optional[Aircraft] = None
        self.steps = 0
        self.last_reward = 0
        
        # Initialize state
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
        self.last_reward = 0
        return self._get_observation()
    
    def step(self, action):
        """Execute one step in the environment."""
        self.steps += 1
        
        # Apply actions
        # Convert relative actions to absolute targets
        target_heading = (self.aircraft.heading + action[0]) % 360
        target_altitude = self.aircraft.altitude + action[1]
        target_speed = self.aircraft.speed + action[2]
        
        # Update aircraft state
        try:
            self.aircraft.update_heading(target_heading)
            self.aircraft.update_altitude(target_altitude)
            self.aircraft.update_speed(target_speed)
            self.aircraft.step()
        except ValueError as e:
            print(f"Warning: Invalid action caused error: {e}")
            return self._get_observation(), -100, True, {}
        
        # Get new state
        observation = self._get_observation()
        
        # Calculate reward
        reward = self._calculate_reward()
        self.last_reward = reward
        
        # Check if episode is done
        done = self._is_done()
        
        return observation, reward, done, {}
    
    def render(self, mode='human'):
        """Render the current environment state."""
        if self.viewer is None:
            from .rendering import ATCRenderer
            self.viewer = ATCRenderer()
        
        # Prepare state for renderer
        env_state = {
            'config': self.config,
            'aircraft': self.aircraft,
            'steps': self.steps,
            'reward': self.last_reward
        }
        
        return self.viewer.render(env_state, mode)
    
    def close(self):
        """Clean up environment resources."""
        if self.viewer:
            self.viewer.close()
            self.viewer = None

