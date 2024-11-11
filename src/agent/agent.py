from collections import defaultdict
import gymnasium as gym
import numpy as np
from torch import nn
from torch import distributions
from src.components import ACTOR_LR
from src.components import CRITIC_LR
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
    def __init__(self, hidden_sizes = (256, 256), max_acc = 3):
        super().__init__()
        self.obs_dim = 12
        self.act_dim = 3
        self.max_acc = max_acc

        self.state_normalizer = RunningNormalizer(self.obs_dim)
        self.actor = nn.Sequential(
            layer_init(nn.Linear(self.obs_dim, hidden_sizes[0])),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_sizes[0], hidden_sizes[1])),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_sizes[1], self.act_dim), std = 0.01),
            nn.Tanh()
        )
        
        initial_std_log = torch.tensor([-1.0, -1.0, -1.0])
        self.policy_std_log = nn.Parameter(initial_std_log.expand(1, self.act_dim))
                
        self.critic = nn.Sequential(
            layer_init(nn.Linear(self.obs_dim, hidden_sizes[0])),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_sizes[0], hidden_sizes[1])),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_sizes[1], 1), std = 1.0)
        )

        self.actor_optimizer = torch.optim.Adam(
            list(self.actor.parameters()) + [self.policy_std_log], 
            lr=ACTOR_LR, eps=1e-5
        )

        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=CRITIC_LR, eps=1e-5
        )

    def train(self, mode=True):
        super().train(mode)
        if mode:
            self.state_normalizer.train()
        else:
            self.state_normalizer.eval()

    def eval(self):
        super().eval()
        self.state_normalizer.eval()

    def act(self, obs):
        obs_norm = self.state_normalizer(obs)

        mean = self.actor(obs_norm)
        std = torch.exp(torch.clamp(self.policy_std_log.expand_as(mean),-20,2))
        dist = distributions.Normal(mean, std)
        a = dist.rsample()

        action = torch.zeros(a.shape)
        action[:, 0] = a[:, 0] * 10.0
        action[:, 1] = a[:, 1] * self.max_acc
        action[:, 2] = a[:, 2] * self.max_acc
        log_pi = dist.log_prob(a).sum(axis=-1)
        return action, log_pi, dist

class RunningNormalizer:
    def __init__(self, dim):
        self.dim = dim
        self.mean = torch.zeros(dim)
        self.std = torch.ones(dim)
        self.count = 0
        self.training = True  # Added training flag
        
    def __call__(self, x):
        # Convert input to tensor if needed
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x).float()
            
        # During inference, just normalize with stored statistics
        if not self.training:
            return (x - self.mean.to(x.device)) / (self.std.to(x.device) + 1e-8)
            
        # Update statistics during training
        batch_mean = x.mean(0)
        batch_var = x.var(0, unbiased=False) + 1e-8
        batch_count = x.shape[0]
        
        delta = batch_mean - self.mean.to(x.device)
        tot_count = self.count + batch_count
        
        new_mean = self.mean.to(x.device) + delta * batch_count / tot_count
        m_a = self.std.to(x.device) ** 2 * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + delta ** 2 * self.count * batch_count / tot_count
        new_std = torch.sqrt(m_2 / tot_count)
        
        self.mean = new_mean.cpu()
        self.std = new_std.cpu()
        self.count = tot_count
        
        return (x - new_mean) / (new_std + 1e-8)
    
    def train(self):
        """Set normalizer to training mode"""
        self.training = True
        
    def eval(self):
        """Set normalizer to evaluation mode"""
        self.training = False



'''
class SACAgent:
    def __init__(self, env):
        self.env = env
        self.model = SACModel()
'''
