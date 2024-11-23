from src.env.environment import DiscreteApproach
import gymnasium as gym
from src.agent.agent import SARSAModel
from tqdm import tqdm
from src.utils import *
NUM_ITER=10000

def train():
    env = gym.make('env/Approach-v2')
    agent = SARSAModel(3, 40)
    reward_history = []
    for _ in tqdm(range(NUM_ITER)):
        obs, info = env.reset()
        done = 0
        reward_history.append(info["total_reward"])
        while not done:
            while not env.need_rl():
                obs, reward, done, _, info = env.step(0)
                if done or _:
                    obs, _ = env.reset()
            action = agent.act(obs)
            next_obs, reward, done, _, info = env.step(action)
            agent.update(action, obs, next_obs, reward)
            
            done = done or _
    plot_list(reward_history, 'reward')
    env.close()
    return agent

def eval(model):
    env = gym.make('env/Approach-v2')
    reward_history = []
    for _ in range(500):
        obs, info = env.reset()
        reward_history.append(info["total_reward"])
        done = 0
        while not done:
            while not env.need_rl():
                obs, reward, done, _, info = env.step(0)
                env.render()
                if done or _:
                    obs, _ = env.reset()
                env.render()
            action = model.act(obs)
            obs, reward, done, truncated, info = env.step(action)
            done = done or truncated
            # 渲染环境（可选）
            env.render()
    plot_list(reward_history, 'eval')
    env.close()
    return

if __name__ == "__main__":
    m = train()
    eval(m)