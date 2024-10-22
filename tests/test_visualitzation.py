import gym
import time
import numpy as np

def test_visualization():
    """Test the environment visualization with manual control."""
    env = gym.make('AtcEnv-v0')
    obs = env.reset()
    
    for _ in range(200):
        # Random action
        action = env.action_space.sample()
        
        # Or test specific maneuver (e.g., turn right and descend)
        # action = np.array([5, -500, 0])  # 5° right, -500ft, no speed change
        
        obs, reward, done, _ = env.step(action)
        env.render()
        time.sleep(0.05)  # Slow down visualization
        
        if done:
            obs = env.reset()
    
    env.close()

if __name__ == "__main__":
    test_visualization()