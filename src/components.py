import gymnasium as gym
from src.env.environment import DummyEnv
from src.agent.agent import SACModel
from src.utils import obs_to_Tensor
import torch

def rollout(agent, env, device, ROLLOUT_STEPS = 128, NUM_ENVS = 1):
    obs, _ = env.reset()
    state = obs_to_Tensor(obs)

    states = torch.zeros((ROLLOUT_STEPS, NUM_ENVS) + env.single_observation_space.shape).to(device)
    actions = torch.zeros((ROLLOUT_STEPS, NUM_ENVS) + env.single_action_space.shape).to(device)
    rewards = torch.zeros((ROLLOUT_STEPS, NUM_ENVS)).to(device)
    dones = torch.zeros((ROLLOUT_STEPS, NUM_ENVS)).to(device)

    logprobs = torch.zeros((ROLLOUT_STEPS, NUM_ENVS)).to(device)
    values = torch.zeros((ROLLOUT_STEPS, NUM_ENVS)).to(device)

    for step in range(0, ROLLOUT_STEPS):
        global_step += NUM_ENVS
        states[step] = state
        dones[step] = done

        with torch.no_grad():
            # Get action, log probability, and entropy from the agent
            action, log_probability = agent.pi(state)
            #value = agent.get_value(state)
            #values[step] = value.flatten()

        actions[step] = action
        logprobs[step] = log_probability

        # Execute action in the environment
        next_state, reward, done, _, info = env.step(action.cpu().numpy())
        rewards[step] = torch.tensor(reward).to(device).view(-1)
        state = torch.Tensor(next_state).to(device)
        done = torch.Tensor(done).to(device)

        if "final_info" in info:
            for episode_info in info["final_info"]:
                if episode_info and "episode" in episode_info:
                    episodic_reward = episode_info['episode']['r']
                    reward_history.append(episodic_reward)
                    episode_history.append(global_step)
