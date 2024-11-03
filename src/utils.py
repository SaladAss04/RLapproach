import torch

def obs_to_Tensor(data):
    agent_speed = torch.Tensor(data["agent"]["speed"])  # 二维向量
    agent_heading = torch.Tensor([data["agent"]["heading"]])  # 标量，转换为 1 维张量
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