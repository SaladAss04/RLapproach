import gymnasium as gym
from src.env.environment import DummyEnv
from src.agent.agent import PPOModel
from src.utils import *
import torch
import numpy as np

ROLLOUT_STEPS = 128 
NUM_ENVS = 1
GAMMA = 0.9
GAE_LAMBDA = 0.95
BATCH_SIZE = 8
NUM_MINI_BATCHES = 4
NUM_EPOCHS = 5
CLIP_COEFF = 0.3
VALUE_LOSS_COEF = 0.5
ENTROPY_COEF = 0.01
NUM_ITERTAIONS = 2

BATCH_SIZE = ROLLOUT_STEPS * NUM_ENVS
MINI_BATCH_SIZE = BATCH_SIZE // NUM_MINI_BATCHES

def rollout(agent, env, device):
    obs, _ = env.reset()
    state = obs_to_Tensor(obs)

    states = torch.zeros((ROLLOUT_STEPS, NUM_ENVS) + state.shape).to(device)
    actions = torch.zeros((ROLLOUT_STEPS, NUM_ENVS) + env.action_space.shape).to(device)
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
            action, log_probability = agent.act(state)
            value = agent.get_value(state)
            values[step] = value.flatten()

        actions[step] = action
        logprobs[step] = log_probability

        # Execute action in the environment
        next_state, reward, done, _, info = env.step(action.cpu().numpy())
        done = done or _
        rewards[step] = torch.tensor(reward).to(device).view(-1)
        state = torch.Tensor(next_state).to(device)
        done = torch.Tensor(done).to(device)
        #?: reset if done
        
    return states, actions, logprobs, rewards, dones, values

def returns(agent, env, states, actions, rewards, dones, values, device):
    with torch.no_grad():
        advantages = torch.zeros_like(rewards).to(device)

        last_gae_lambda = 0
        for t in reversed(range(ROLLOUT_STEPS)):
            if t == ROLLOUT_STEPS - 1:
                next_non_terminal = 1.0 - dones[-1]
                next_value = agent.get_value(states[-1]).reshape(1, -1)
            else:
                next_non_terminal = 1.0 - dones[t + 1]
                next_value = values[t + 1]

            # Compute delta using the utility function
            delta = get_deltas(rewards[t], values[t], next_value, next_non_terminal, gamma=GAMMA)

            advantages[t] = last_gae_lambda = delta + GAMMA * GAE_LAMBDA * next_non_terminal * last_gae_lambda

    return advantages

def train(agent, env, states, actions, logprobs, rewards, dones, values, advantages, device):
    batch_states = states.reshape((-1,) + env.single_observation_space.shape)
    batch_logprobs = logprobs.reshape(-1)
    batch_actions = actions.reshape((-1,) + env.single_action_space.shape)
    batch_advantages = advantages.reshape(-1)
    batch_returns = returns.reshape(-1)
    batch_values = values.reshape(-1)

    # Shuffle the batch data to break correlation between samples
    batch_indices = np.arange(BATCH_SIZE)
    total_actor_loss = 0
    total_critic_loss = 0
    total_entropy_objective = 0
    
    for epoch in range(NUM_EPOCHS):
        np.random.shuffle(batch_indices)
        for start in range(0, BATCH_SIZE, MINI_BATCH_SIZE):
            # Get the indices for the mini-batch
            end = start + MINI_BATCH_SIZE
            mini_batch_indices = batch_indices[start:end]

            mini_batch_advantages = batch_advantages[mini_batch_indices]
            # Normalize advantages to stabilize training
            mini_batch_advantages = (mini_batch_advantages - mini_batch_advantages.mean()) / (mini_batch_advantages.std() + 1e-8)

            # Compute new probabilities and values for the mini-batch
            new_probabilities = agent.get_probs(batch_states[mini_batch_indices])
            new_log_probability = agent.get_action_logprob(new_probabilities, batch_actions.long()[mini_batch_indices])
            entropy = agent.get_entropy(new_probabilities)
            new_value = agent.get_value(batch_states[mini_batch_indices])

            # Calculate the policy loss
            ratio = get_ratio(new_log_probability, batch_logprobs[mini_batch_indices])
            policy_objective = get_policy_objective(mini_batch_advantages, ratio, clip_coeff=CLIP_COEFF)
            policy_loss = -policy_objective

            # Calculate the value loss
            value_loss = get_value_loss(new_value.view(-1), batch_values[mini_batch_indices], batch_returns[mini_batch_indices], clip_coeff=CLIP_COEFF)

            # Calculate the entropy loss
            entropy_objective = entropy.mean()

            # Combine losses to get the total loss
            total_loss = get_total_loss(policy_objective, value_loss, entropy_objective, value_loss_coeff=VALUE_LOSS_COEF, entropy_coeff=ENTROPY_COEF)

            optimizer = torch.optim.Adam(agent.parameters(), eps=1e-5)
            
            optimizer.zero_grad()
            total_loss.backward()
            # Clip the gradient to stabilize training
            torch.nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
            optimizer.step()

            total_actor_loss += policy_loss.item()
            total_critic_loss += value_loss.item()
            total_entropy_objective += entropy_objective.item()
