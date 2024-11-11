import numpy as np
from src.atc_env import ATCplanning

def test_environment():
    """
    Test suite for ATCplanning environment
    """
    env = ATCplanning()
    print("\n=== Testing Environment ===")

    # 1. Test Reset
    print("\n1. Testing Reset...")
    obs, info = env.reset()
    
    print("Initial State:")
    print(f"Agent Position: {obs['agent']['position']}")
    print(f"Agent Heading: {obs['agent']['heading']}")
    print(f"Agent Altitude: {obs['agent']['altitude']}")
    print(f"Agent Speed: {obs['agent']['speed']}")
    print(f"Initial Info: {info}")

    # 2. Test Step with different actions
    print("\n2. Testing Steps...")
    
    # Test case 2.1: Straight flight
    print("\nTest 2.1: Straight flight")
    action = {
        'turn': np.array([0.0]),
        'change_horizontal_acceleration': np.array([0.0]),
        'change_vertical_acceleration': np.array([0.0])
    }
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"Reward: {reward}")
    print(f"New Position: {obs['agent']['position']}")
    print(f"Speed: {obs['agent']['speed']}")

    # Test case 2.2: Turn right
    print("\nTest 2.2: Turn right")
    action = {
        'turn': np.array([10.0]),  # Maximum turn rate
        'change_horizontal_acceleration': np.array([0.0]),
        'change_vertical_acceleration': np.array([0.0])
    }
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"New Heading: {obs['agent']['heading']}")

    # Test case 2.3: Accelerate
    print("\nTest 2.3: Accelerate")
    action = {
        'turn': np.array([0.0]),
        'change_horizontal_acceleration': np.array([3.0]),  # Max acceleration
        'change_vertical_acceleration': np.array([0.0])
    }
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"New Speed: {obs['agent']['speed']}")

    # Test case 2.4: Change altitude
    print("\nTest 2.4: Change altitude")
    action = {
        'turn': np.array([0.0]),
        'change_horizontal_acceleration': np.array([0.0]),
        'change_vertical_acceleration': np.array([2.0])
    }
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"New Altitude: {obs['agent']['altitude']}")

    # 3. Test Boundaries
    print("\n3. Testing Boundaries...")
    
    # Test case 3.1: Try to go below minimum altitude
    print("\nTest 3.1: Testing minimum altitude boundary")
    action = {
        'turn': np.array([0.0]),
        'change_horizontal_acceleration': np.array([0.0]),
        'change_vertical_acceleration': np.array([-10.0])  # Strong downward
    }
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"Altitude after attempting to go too low: {obs['agent']['altitude']}")
    print(f"Reward (should be negative due to min altitude violation): {reward}")

    # Test case 3.2: Try to exceed maximum speed
    print("\nTest 3.2: Testing maximum speed boundary")
    action = {
        'turn': np.array([0.0]),
        'change_horizontal_acceleration': np.array([10.0]),  # Try to accelerate too much
        'change_vertical_acceleration': np.array([0.0])
    }
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"Speed after attempting to exceed max speed: {obs['agent']['speed']}")

    # 4. Test Episode Termination
    print("\n4. Testing Episode Termination...")
    for i in range(10):  # Run for a few steps
        action = {
            'turn': np.array([0.0]),
            'change_horizontal_acceleration': np.array([0.0]),
            'change_vertical_acceleration': np.array([0.0])
        }
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            print(f"Episode terminated after {i+1} steps")
            print(f"Terminated: {terminated}, Truncated: {truncated}")
            break

    # 5. Test Reward Function
    print("\n5. Testing Reward Function...")
    env.reset()
    rewards = []
    for _ in range(5):
        action = {
            'turn': np.array([0.0]),
            'change_horizontal_acceleration': np.array([1.0]),
            'change_vertical_acceleration': np.array([0.0])
        }
        _, reward, _, _, _ = env.step(action)
        rewards.append(reward)
    print(f"Sequence of rewards: {rewards}")

    print("\n=== Tests Complete ===")

if __name__ == "__main__":
    test_environment()
