from src.agent.agent import PPOModel
import gymnasium as gym
from src.components import *
from src.env.environment import DummyEnv
from src.env.atc_env import ATCplanning
import torch
import numpy as np
from src.utils import obs_to_Tensor

def make_env():
    #return gym.make('env/Approach-v0')
    return gym.make('env/Approach-v1')
    #return gym.make('env/Approach-v2')

def test_trained_model(model_path, num_episodes=5, render=True):
    """
    Test a trained model and render its behavior
    """
    # Create environment
    env = make_env()
    
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = PPOModel()
    
    # Load the checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Extract the model state dict
    if "model_state_dict" in checkpoint:
        agent.load_state_dict(checkpoint['model_state_dict'])
    else:
        agent.load_state_dict(checkpoint)
    
    agent.to(device)
    agent.eval()  # Set to evaluation mode

    total_rewards = []
    success_count = 0

    for episode in range(num_episodes):
        obs, _ = env.reset()
        state = obs_to_Tensor(obs).to(device)
        episode_reward = 0
        done = False
        steps = 0
        
        print(f"\nEpisode {episode + 1}")
        print("-" * 20)

        while not done:
            # Get action from model
            with torch.no_grad():
                action, _, _ = agent.act(state)
                action_np = action.cpu().numpy().flatten()

            # Take step in environment
            next_obs, reward, terminated, truncated, info = env.step(action_np)
            done = terminated or truncated
            
            if render:
                env.render()
            
            # Update for next iteration
            state = obs_to_Tensor(next_obs).to(device)
            episode_reward += float(reward)  # Convert reward to float
            steps += 1

            # Print step information with proper formatting
            print(f"Step {steps}: Action=[{', '.join(f'{x:.2f}' for x in action_np)}], Reward={float(reward):.1f}")

        # Episode summary
        total_rewards.append(episode_reward)
        if episode_reward > 0:  # Assuming positive reward means success
            success_count += 1
        
        print(f"\nEpisode Summary:")
        print(f"Total Reward: {float(episode_reward):.1f}")
        print(f"Steps Taken: {steps}")
        print(f"Episode Result: {'Success' if episode_reward > 0 else 'Failure'}")

    # Overall performance
    print("\nOverall Performance:")
    print(f"Average Reward: {float(np.mean(total_rewards)):.1f}")
    print(f"Success Rate: {success_count/num_episodes:.0%}")

    env.close()

if __name__ == "__main__":
    model_path = "policy.pth"  # or "best_model.pth" or specific checkpoint
    test_trained_model(model_path, num_episodes=5, render=True)
