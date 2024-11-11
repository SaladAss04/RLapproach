import torch
import matplotlib.pyplot as plt
import numpy as np

def obs_to_Tensor(data):
    
    def process_speed(speed):
        # Ensure speed is a 2D vector
        if isinstance(speed, (float, np.float64, np.float32)):
            return torch.FloatTensor([speed, 0.0])
        return torch.FloatTensor(speed)

    def process_scalar(value):
        # Convert scalar values to single-element tensors
        return torch.FloatTensor([float(value)])

    def process_position(pos):
        # Ensure position is a 2D vector
        return torch.FloatTensor(pos)

    # Check if we're dealing with a single environment observation
    is_single = not isinstance(data["agent"]["speed"], np.ndarray) or data["agent"]["speed"].ndim == 1

    if is_single:
        # Process single environment case
        agent_speed = process_speed(data["agent"]["speed"])
        agent_heading = process_scalar(data["agent"]["heading"])
        agent_altitude = process_scalar(data["agent"]["altitude"])
        agent_position = process_position(data["agent"]["position"])
        
        target_speed = process_speed(data["target"]["speed"])
        target_heading = process_scalar(data["target"]["heading"])
        target_altitude = process_scalar(data["target"]["altitude"])
        target_position = process_position(data["target"]["position"])

        return torch.cat([
            agent_speed.flatten(),  # [2]
            agent_heading.flatten(),  # [1]
            agent_altitude.flatten(),  # [1]
            agent_position.flatten(),  # [2]
            target_speed.flatten(),  # [2]
            target_heading.flatten(),  # [1]
            target_altitude.flatten(),  # [1]
            target_position.flatten()  # [2]
        ]).unsqueeze(0)  # Add batch dimension
    
    # Handle vectorized environment case
    num_envs = len(data["agent"]["speed"]) if isinstance(data["agent"]["speed"], np.ndarray) else 1
    result = torch.zeros((num_envs, 12))
    
    for i in range(num_envs):
        agent_speed = process_speed(data["agent"]["speed"][i])
        agent_heading = process_scalar(data["agent"]["heading"][i])
        agent_altitude = process_scalar(data["agent"]["altitude"][i])
        agent_position = process_position(data["agent"]["position"][i])
        
        target_speed = process_speed(data["target"]["speed"][i])
        target_heading = process_scalar(data["target"]["heading"][i])
        target_altitude = process_scalar(data["target"]["altitude"][i])
        target_position = process_position(data["target"]["position"][i])

        # Concatenate all features
        line = torch.cat([
            agent_speed.flatten(),  # [2]
            agent_heading.flatten(),  # [1]
            agent_altitude.flatten(),  # [1]
            agent_position.flatten(),  # [2]
            target_speed.flatten(),  # [2]
            target_heading.flatten(),  # [1]
            target_altitude.flatten(),  # [1]
            target_position.flatten()  # [2]
        ])
        result[i] = line
    
    return result

def get_deltas(rewards, values, next_values, next_nonterminal, gamma):
    rewards = rewards.squeeze()
    values = values.squeeze()
    next_values = next_values.squeeze()
    deltas = torch.zeros_like(rewards)
    deltas[next_nonterminal == 1] = rewards[next_nonterminal == 1] + gamma * next_values[next_nonterminal == 1] - values[next_nonterminal == 1]
    deltas[next_nonterminal != 1] = rewards[next_nonterminal != 1] - values[next_nonterminal != 1]

    return deltas

def get_ratio(logprob, logprob_old):
    logratio = logprob - logprob_old  
    ratio = torch.exp(logratio)  
    return ratio
    
def get_policy_objective(advantages, ratio, clip_coeff):
    policy_objective1 = ratio * advantages  
    policy_objective2 = torch.clamp(ratio, 1 - clip_coeff, 1 + clip_coeff) * advantages  
    policy_objective = torch.min(policy_objective1, policy_objective2) 
    policy_objective = torch.mean(policy_objective)
    return policy_objective

def get_value_loss(values, values_old, returns, clip_coeff):
    value_loss_unclipped = 0.5 * (values - returns) * (values - returns) # Calculate unclipped value loss
    value_loss_clipped = 0.5 * (values_old + torch.clamp(values - values_old, -clip_coeff, clip_coeff) - returns) * (values_old + torch.clamp(values - values_old, -clip_coeff, clip_coeff) - returns)
    value_loss = torch.mean(torch.max(value_loss_clipped, value_loss_unclipped))  # Average over the batch
    
    return value_loss

def get_total_loss(policy_objective, value_loss, entropy_objective, value_loss_coeff, entropy_coeff):
    total_loss = -policy_objective + value_loss_coeff * value_loss - entropy_coeff * entropy_objective  # Combine losses
    return total_loss

def plot(x, y, label):
    plt.figure(figsize=(10, 5))
    plt.plot(x, y, label=label, alpha=0.5)
    plt.xlabel("Iteration")
    plt.ylabel("Value")
    plt.savefig(label + '.png')

#def plot_set(dict):
#    for key in dict.keys():
#        plot(range(len(dict[key])), dict[key], key)

def plot_set(dict):
    for key in dict.keys():
        if key == 'eval_reward':
            plt.figure(figsize=(10,5))
            plt.plot(range(0, len(dict['reward']), EVAL_FREQUENCY),
                    dict['eval_reward'],
                    label='evaluation_reward',
                    alpha=0.5)
            plt.xlabel("Iteration")
            plt.ylabel("Value")
            plt.savefig('evaluation_reward.png')
            plt.close()
        else:
            try:
                # Convert to numpy array and flatten if needed
                data = np.array(dict[key]).flatten()
                plt.figure(figsize=(10, 5))
                plt.plot(range(len(data)), data, label=key, alpha=0.5)
                plt.xlabel("Iteration")
                plt.ylabel("Value")
                plt.savefig(f'{key}.png')
                plt.close()
            except Exception as e:
                print(f"Could not plot {key}: {e}")
