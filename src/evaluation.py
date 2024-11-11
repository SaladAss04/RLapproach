import torch
import numpy as np
from src.utils import obs_to_Tensor

def evaluate_policy(agent, env, device, num_episodes=20):
    total_rewards = []
    successes = 0
    #metrics = {
    #    'heading_diff': [],
    #    'altitude_diff': [],
    #    'speed_diff': [],
    #    'episode_len': []
    #}

    with torch.no_grad():
        for _ in range(num_episodes):
            obs, _ = env.reset()
            state = obs_to_Tensor(obs).to(device)
            done = False
            episode_reward = 0
            while not done:
                action, _, _ = agent.act(state)
                #convert action to numpy and ensure correct shape
                action_np = action.cpu().numpy().flatten()  #flatten to 1D array
                next_obs, reward, terminated, truncated, _ = env.step(action_np)
                done = terminated or truncated
                episode_reward += reward


                if terminated and reward > 0:
                    successes += 1

                # if done:
                #     metrics['heading_diff'].append(info['heading_difference'])
                #     metrics['altitude_diff'].append(info['altitude_difference'])
                #     metrics['speed_diff'].append(info['speed_difference'])
                #     metrics['episode_lengths'].append(info['total_steps'])
                #     if terminated and reward > 0:  #successful termination
                #         successes += 1
                
                if not done:
                    state = obs_to_Tensor(next_obs).to(device)

            total_rewards.append(episode_reward)

    success_rate = successes / num_episodes
    return np.mean(total_rewards), success_rate 
