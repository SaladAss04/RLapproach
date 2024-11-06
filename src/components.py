from src.utils import *
import torch
import numpy as np

NUM_ENVS = 3
ROLLOUT_STEPS = 1024
GAMMA = 0.9
GAE_LAMBDA = 0.95
BATCH_SIZE = 8
NUM_MINI_BATCHES = 4
NUM_EPOCHS = 5
CLIP_COEFF = 0.3
VALUE_LOSS_COEF = 0.5
ENTROPY_COEF = 0.01
BATCH_SIZE = ROLLOUT_STEPS * NUM_ENVS
MINI_BATCH_SIZE = BATCH_SIZE // NUM_MINI_BATCHES

def rollout(agent, env, device):
    '''
    The rollout step lets the current agent interact with the environment, records quantities needed for the 
    RL, and records its performance for logging/visualizatiob.
    '''
    obs, _ = env.reset()
    state = obs_to_Tensor(obs)
    states = torch.zeros((ROLLOUT_STEPS, NUM_ENVS, state.shape[1])).to(device)
    actions = torch.zeros((ROLLOUT_STEPS, NUM_ENVS) + env.action_space.shape).to(device)
    rewards = torch.zeros((ROLLOUT_STEPS, NUM_ENVS)).to(device)
    dones = torch.zeros((ROLLOUT_STEPS, NUM_ENVS)).to(device)
    done = torch.zeros((NUM_ENVS,)).to(device)
    
    reward_history = []
    episodic_history = []
    
    logprobs = torch.zeros((ROLLOUT_STEPS, NUM_ENVS)).to(device)
    values = torch.zeros((ROLLOUT_STEPS, NUM_ENVS)).to(device)

    for step in range(0, ROLLOUT_STEPS):
        states[step] = state
        dones[step] = done

        with torch.no_grad():
            # Get action, log probability, and entropy from the agent
            action, log_probability, _ = agent.act(state)
            value = agent.critic(state)
            values[step] = value.flatten()

        actions[step] = action
        logprobs[step] = log_probability

        # Execute action in the environment
        next_state, reward, done, _, info = env.step(action.cpu().numpy())
        done = [done[i] or _[i] for i in range(len(done))]
        rewards[step] = torch.tensor(reward).to(device).view(-1)
        state = torch.Tensor(obs_to_Tensor(next_state)).to(device)
        done = torch.Tensor(done).to(device)
        if done.any():
            reward_history.extend(info['total_reward'][done == 1])
            episodic_history.extend(info['total_steps'][done == 1])
        
    return states, actions, logprobs, rewards, dones, values, reward_history, episodic_history

def returns(agent, states, actions, rewards, dones, values, device):
    '''
    The returns step calculates advantage function out of rollout results, so as to calculate loss/objectives needed for training.
    '''
    with torch.no_grad():
        advantages = torch.zeros_like(rewards).to(device)

        last_gae_lambda = 0
        for t in reversed(range(ROLLOUT_STEPS)):
            if t == ROLLOUT_STEPS - 1:
                next_non_terminal = 1.0 - dones[-1]
                next_value = agent.critic(states[-1])
            else:
                next_non_terminal = 1.0 - dones[t + 1]
                next_value = values[t + 1]

            # Compute delta using the utility function
            delta = get_deltas(rewards[t], values[t], next_value, next_non_terminal, gamma=GAMMA)

            advantages[t] = last_gae_lambda = delta + GAMMA * GAE_LAMBDA * next_non_terminal * last_gae_lambda

    return advantages

def train(agent, env, states, actions, logprobs, values, advantages):
    '''
    Excecuted and updates parameters PER ROLLOUT.
    '''
    returns = advantages + values
    
    batch_states = states.reshape((-1, states.shape[-1]))
    batch_logprobs = logprobs.reshape(-1)
    batch_actions = actions.reshape((-1,) + env.action_space.shape)
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
            _, n_logprobs, n_probs = agent.act(batch_states[mini_batch_indices])
            entropy = n_probs.entropy()
            new_value = agent.critic(batch_states[mini_batch_indices])

            # Calculate the policy loss
            ratio = get_ratio(n_logprobs, batch_logprobs[mini_batch_indices])
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
            
    return total_actor_loss, total_critic_loss, total_entropy_objective    
    
