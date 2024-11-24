from collections import defaultdict
import gymnasium as gym
import numpy as np
from torch import nn
from torch.distributions.categorical import Categorical
import torch
import random

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

class SARSAModel(nn.Module):
    def __init__(self, num_obstacles, size):
        super().__init__()
        self.obs_dim = num_obstacles
        self.act_dim = 5
        self.size = size
        #self.q_table = np.zeros((self.size, 36, self.size, 36, 36, self.act_dim)) #Only decide based on the closest obstacle, and where the target is at
        self.q_table = np.random.uniform(low=-50, high=50, size=(self.size, 36, self.size, 36, 36, self.act_dim))
    
    def act(self, obs, epsilon = 0.2, episode = None):
        if episode:
            epsilon = max(0.15, epsilon - 1e-5 * episode)
        i = np.argmin(obs[:-1, 0])
        #print("avoiding obstacle", i)
            
        q_values = self.q_table[obs[-1][0], obs[-1][1], obs[i][0], obs[i][1], obs[i][2], :]
        greedy_action = np.argmax(q_values)
        if random.uniform(0, 1) < epsilon:
            rest_actions = [x for x in range(self.act_dim) if x != greedy_action]
            action = rest_actions[random.randint(0, len(rest_actions) - 1)]
        else:
            action = greedy_action
        return action

    def update(self, action, obs_old, obs_new, reward, gamma=0.9, alpha=0.8, episode=None):
        if episode is not None:
            alpha = max(alpha - 5e-5 * episode, 0.4)
        i = np.argmin(obs_old[:-1, 0]) #focus on the original closest obstacle
        next_action = self.act(obs_new)
        next_q = self.q_table[obs_new[-1][0], obs_new[-1][1], obs_new[i][0], obs_new[i][1], obs_new[i][2], next_action]
        old_q = self.q_table[obs_old[-1][0], obs_old[-1][1], obs_old[i][0], obs_old[i][1], obs_old[i][2], action]
        increment = reward + gamma * next_q - old_q
        self.q_table[obs_old[-1][0], obs_old[-1][1], obs_old[i][0], obs_old[i][1], obs_old[i][2], action] += alpha * increment
        return 
