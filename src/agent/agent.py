from collections import defaultdict
import gymnasium as gym
import numpy as np
from torch import nn
from torch.distributions.categorical import Categorical
import torch

class DummyAgent:
    def __init__(self,
                env         
                ) -> None:
        self.env = env
    
    def get_action(self, obs: dict) -> dict:
        '''
        observation is the tuple of two dictionaries: agent_state and target_state,
            each: {'position', 'altitude', 'heading', 'speed'}
        action (output) is a dictionary: {'turn', 'horizontal_acceleration', 'vertical_acceleration'}
        '''
        agent, target = obs['agent'], obs['target']
        assert agent['position'].dim == 2 and agent['position'].dtype == np.float16
        assert target['position'].dim == 2 and target['position'].dtype == np.float16
        assert agent['speed'].dim == 2 and agent['speed'].dtype == np.float16
        assert target['speed'].dim == 2 and target['speed'].dtype == np.float16
        assert isinstance(agent['heading']) and 0 <= agent['heading'] <= 360
        assert isinstance(target['heading']) and 0 <= target['heading'] <= 360
        
        action =  {'turn': np.random.random_integers(0, 360, size=1, dtype=int),
                'h_acc': self.env.rng.uniform(-self.env.max_acc, -self.env.max_acc, size=1, dtype=np.float16),
                'v_acc': self.env.rng.uniform(-self.env.max_acc, -self.env.max_acc, size=1, dtype=np.float16)}
        
        acc = np.array(action['h_acc'], action['v_acc'], dtype=np.float16)
        norm_v = np.linalg.norm(acc, ord=2)
        if norm_v > self.max_acc:
            acc = acc * (self.max_acc / norm_v)
            action['h_acc'] = acc[0]
            action['v_acc'] = acc[1]
        
        return action
    
    def update(self):
        pass

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """Initialize the weights and biases of a layer.
    
    Args:
        layer (nn.Module): The layer to initialize.
        std (float): Standard deviation for orthogonal initialization.
        bias_const (float): Constant value for bias initialization.
    
    Returns:
        nn.Module: The initialized layer.
    """
    nn.init.orthogonal_(layer.weight, std)  # Orthogonal initialization
    nn.init.constant_(layer.bias, bias_const)  # Constant bias
    return layer

class PPOModel(nn.Module):
    def __init__(self, num_obstacles, hidden_sizes = (256, 256)):
        super().__init__()
        self.obs_dim = 3 * (num_obstacles + 1)
        self.act_dim = 5
        self.actor = nn.Sequential(
            layer_init(nn.Linear(self.obs_dim, hidden_sizes[0])),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_sizes[0], hidden_sizes[1])),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_sizes[1], self.act_dim), std = 0.01),
            nn.Softmax(dim = -1)
        )
        
        self.policy_std_log = nn.Parameter(torch.zeros(self.act_dim))
        
        self.critic = nn.Sequential(
            layer_init(nn.Linear(self.obs_dim, hidden_sizes[0])),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_sizes[0], hidden_sizes[1])),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_sizes[1], 1), std = 0.1)
        )
        
    def act(self, obs):
        logits = self.actor(obs)
        dist = Categorical(logits)
        a = dist.sample()
        log_pi = dist.log_prob(a)
        return a, log_pi, dist
