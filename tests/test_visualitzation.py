import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import gym
import time
import numpy as np

# Import the environment to ensure it's registered
import envs

def test_visualization():
    """Test the environment visualization with manual control."""
    env = gym.make('AtcEnv-v0')
    obs = env.reset()
    
    print("Starting visualization test...")
    print("Initial observation:", obs)

    try:
        for i in range(200):
            # Random action
            action = env.action_space.sample()
            
            # Or test specific maneuver (e.g., turn right and descend)
            # action = np.array([5, -500, 0])  # 5° right, -500ft, no speed change
            
            obs, reward, done, _ = env.step(action)
            env.render()
            time.sleep(0.05)  # Slow down visualization
        
            if i % 20 == 0:
                print(f"Step {i}, Reward: {reward}")
        if done:
            print("Episode finished after", i, "steps")
            obs = env.reset()

    except Exception as e:
        print("Error during visualization:", e)
        raise e
    
    finally:
        env.close()

if __name__ == "__main__":
    test_visualization()