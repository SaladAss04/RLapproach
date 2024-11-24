from src.env.environment import DiscreteApproach
import gymnasium as gym
from src.agent.agent import SARSAModel
from tqdm import tqdm
from src.utils import *
import numpy as np
NUM_ITER=500

def train():
    env = gym.make('env/Approach-v2')
    agent = SARSAModel(6, env.max_dis)
    reward_history = []
    reward_mean = []
    for t in tqdm(range(NUM_ITER)):
        obs, info = env.reset()
        done = 0
        reward_history.append(info["total_reward"])
        reward_mean.append(np.mean(reward_history))
        while not done:
            while not env.need_rl():
                obs, reward, done, truncated, info = env.direct_step()
                a, p, h, diff = env.get_position_debug()
                #print(a, p, h, diff)
                if done or truncated:
                    obs, info = env.reset()
                    #print("straight to target, restarting")
                #env.render()
            action = agent.act(obs, episode=t)
            next_obs, reward, done, truncated, info = env.step(action)
            agent.update(action, obs, next_obs, reward)
            
            done = done or truncated
    plot_list(reward_history, 'reward')
    plot_list(reward_mean, 'reward mean')
    env.close()
    return agent

def eval(model):
    env = gym.make('env/Approach-v2')
    reward_history = []
    for t in range(500):
        obs, info = env.reset()
        reward_history.append(info["total_reward"])
        done = 0
        while not done:
            while not env.need_rl():
                obs, reward, done, truncated, info = env.direct_step()
                env.render()
                if done or truncated:
                    obs, info = env.reset()
                env.render()
            '''
            if obs[-1][0] >= 40:
                print(env.get_position_debug())
            else:
                print(obs[-1][0])
            '''
            action = model.act(obs, epsilon=0.08)
            obs, reward, done, truncated, info = env.step(action)
            done = done or truncated
            env.render()
    plot_list(reward_history, 'eval')
    env.close()
    return

if __name__ == "__main__":
    m = train()
    eval(m)