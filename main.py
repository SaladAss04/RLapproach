from src.agent.agent import PPOModel
import gymnasium as gym
from src.components import *
from src.components import NUM_ENVS
from src.env.environment import DummyEnv
import torch
from tqdm import tqdm

NUM_ITERTAIONS = 5
def make_env():
    return gym.make('env/Approach-v0')

def create_parallel_envs(num = NUM_ENVS):
    ret = gym.vector.SyncVectorEnv([make_env for _ in range(num)])
    return ret

def main():
    envs = create_parallel_envs() 
    agent = PPOModel()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for _ in tqdm(range(NUM_ITERTAIONS)):
        states, actions, logprobs, rewards, dones, values = rollout(agent, envs, device)
        advantage = returns(agent, states, actions, rewards, dones, values, device)
        train(agent, envs, states, actions, logprobs, values, advantage)
    envs.close()
    torch.save(agent.state_dict(), 'model.pth')
    
if __name__ == "__main__":
    main()