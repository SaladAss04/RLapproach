import torch

def obs_to_Tensor(data):
    agent_speed = torch.Tensor(data["agent"]["speed"])  
    agent_heading = torch.Tensor([data["agent"]["heading"]])  
    agent_altitude = torch.Tensor([data["agent"]["altitude"]])  # 标量，转换为 1 维张量
    agent_position = torch.Tensor(data["agent"]["position"])  # 二维向量

    target_speed = torch.Tensor(data["target"]["speed"])  # 二维向量
    target_heading = torch.Tensor([data["target"]["heading"]])  # 标量，转换为 1 维张量
    target_altitude = torch.Tensor([data["target"]["altitude"]])  # 标量，转换为 1 维张量
    target_position = torch.Tensor(data["target"]["position"])  # 二维向量

    # 将所有张量拼接成一个 12 维张量
    result = torch.cat([
        agent_speed, agent_heading, agent_altitude, agent_position,
        target_speed, target_heading, target_altitude, target_position
    ])
    
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