import numpy as np
from src.atc_env_rendered import ATCplanning
import time
import pygame

def test_environment():
    # Initialize pygame
    pygame.init()
    
    # Create environment with render_mode specified
    env = ATCplanning(
        size=10000, 
        max_acc=3, 
        max_speed=300, 
    )
    
    # Reset environment
    observation, info = env.reset()
    
    running = True
    step = 0
    
    while running and step < 100:
        # Handle Pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
        
        # Create random action
        action = {
            'turn': float(np.random.uniform(-10, 10)),
            'change_horizontal_acceleration': float(np.random.uniform(-3, 3)),
            'change_vertical_acceleration': float(np.random.uniform(-3, 3))
        }
        
        # Take step in environment
        observation, reward, terminated, truncated, info = env.step(action)
        
        # Render the environment
        env.render()
        
        # Print some info
        print(f"Step {step + 1}")
        print(f"Reward: {reward:.2f}")
        print(f"Distance to target: {info['distance']:.2f}")
        print(f"Altitude: {observation['agent']['altitude']:.2f}")
        print("-" * 50)
        
        # Check if episode is done
        if terminated or truncated:
            print("Episode finished!")
            break
            
        step += 1
        
        # Control the frame rate
        time.sleep(0.1)
    
    # Proper cleanup
    env.close()
    pygame.quit()

if __name__ == "__main__":
    try:
        test_environment()
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
        pygame.quit()
