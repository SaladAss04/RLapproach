from src.agent.agent import PPOModel
import gymnasium as gym
#from src.components import *
from src.utils import *
from src.env.environment import DiscreteApproach
import torch
from torch import nn
from tqdm import tqdm
import numpy as np

NUM_ITERATIONS = 200
NUM_ENVS = 1
ROLLOUT_STEPS = 256
GAMMA = 0.99
GAE_LAMBDA = 0.95
BATCH_SIZE = 8
NUM_MINI_BATCHES = 4
NUM_EPOCHS = 5
CLIP_COEF = 0.2
VALUE_LOSS_COEF = 0.5
ENTROPY_COEF = 0.01
BATCH_SIZE = ROLLOUT_STEPS * NUM_ENVS
MINI_BATCH_SIZE = BATCH_SIZE // NUM_MINI_BATCHES
LEARNING_RATE = 1e-3

def make_env():
    return gym.make('env/Approach-v2')

def create_parallel_envs(num = NUM_ENVS):
    ret = gym.vector.SyncVectorEnv([make_env for _ in range(num)])
    return ret
# Initialize global step counter and reset the environment
def main():
    #envs = create_parallel_envs()
    env = gym.make('env/Approach-v2')
    global_step = 0
    agent = PPOModel(3)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    states = torch.zeros((ROLLOUT_STEPS, NUM_ENVS) + env.observation_space.shape).to(device)
    actions = torch.zeros((ROLLOUT_STEPS, NUM_ENVS) + env.action_space.shape).to(device)
    rewards = torch.zeros((ROLLOUT_STEPS, NUM_ENVS)).to(device)
    dones = torch.zeros((ROLLOUT_STEPS, NUM_ENVS)).to(device)
    logprobs = torch.zeros((ROLLOUT_STEPS, NUM_ENVS)).to(device)
    values = torch.zeros((ROLLOUT_STEPS, NUM_ENVS)).to(device)

    initial_state, _ = env.reset()
    state = torch.Tensor(initial_state).to(device)
    done = torch.zeros(NUM_ENVS).to(device)

    # Set up progress tracking
    progress_bar = tqdm(range(1, NUM_ITERATIONS + 1), postfix={'Total Rewards': 0})
    actor_loss_history = []
    critic_loss_history = []
    entropy_objective_history = []

    reward_history = []
    episode_history = []

    optimizer = torch.optim.Adam(agent.parameters(), lr=LEARNING_RATE, eps=1e-5)

    data_to_plot = {
            'Total Reward': [],
            'Actor Loss': [],
            'Critic Loss': [],
            'Entropy': []
    }
    for iteration in progress_bar:
        # Adjust the learning rate using a linear decay
        fraction_completed = 1.0 - (iteration - 1.0) / NUM_ITERATIONS
        current_learning_rate = fraction_completed * LEARNING_RATE
        optimizer.param_groups[0]["lr"] = current_learning_rate

        # Perform rollout to gather experience
        for step in range(0, ROLLOUT_STEPS):
            while not env.need_rl():
                state, reward, done, _, info = env.step(0)
                #pos, tar, heading, target = env.get_position_debug()
                #print(pos, tar, heading, target)
                #assert heading == target
                env.render()
                if done or _:
                    state, _ = env.reset()
                #print("skipping")
            #print("encounter obstacle", env.get_position_debug()) 
            state = torch.Tensor(initial_state).to(device)
            done = torch.Tensor([done]).to(device) 

            global_step += NUM_ENVS
            states[step] = state
            dones[step] = done

            with torch.no_grad():
                # Get action, log probability, and entropy from the agent
                action, log_probability, _ = agent.act(state.view(NUM_ENVS, -1))
                value = agent.critic(state.view(NUM_ENVS, -1))
                values[step] = value.flatten()

            actions[step] = action
            logprobs[step] = log_probability

            # Execute action in the environment
            next_state, reward, done, _, info = env.step(action.cpu().numpy()[0])
            #normalized_reward = (reward - min_reward) / (max_reward - min_reward)  # Normalize the reward
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            state = torch.Tensor(next_state).to(device)
            done = torch.Tensor([done]).to(device)

            env.render()
            if "final_info" in info:
                for episode_info in info["final_info"]:
                    if episode_info and "episode" in episode_info:
                        episodic_reward = episode_info['episode']['r']
                        reward_history.append(episodic_reward)
                        episode_history.append(global_step)
                        progress_bar.set_postfix({'Total Rewards': episodic_reward})

        # Calculate advantages and returns
        with torch.no_grad():
            next_value = agent.critic(state.view(NUM_ENVS, -1)).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)

            last_gae_lambda = 0
            for t in reversed(range(ROLLOUT_STEPS)):
                if t == ROLLOUT_STEPS - 1:
                    next_non_terminal = 1.0 - done
                    next_value = next_value
                else:
                    next_non_terminal = 1.0 - dones[t + 1]
                    next_value = values[t + 1]

                # Compute delta using the utility function
                delta = get_deltas(rewards[t], values[t], next_value, next_non_terminal, gamma=GAMMA)

                advantages[t] = last_gae_lambda = delta + GAMMA * GAE_LAMBDA * next_non_terminal * last_gae_lambda
            returns = advantages + values

        # Flatten the batch data for processing
        batch_states = states.reshape((-1,) + env.observation_space.shape)
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
                '''
                new_probabilities = agent.get_probs(batch_states[mini_batch_indices])
                new_log_probability = agent.get_action_logprob(new_probabilities, batch_actions.long()[mini_batch_indices])
                entropy = agent.get_entropy(new_probabilities)
                new_value = agent.critic(batch_states[mini_batch_indices])
                '''
                _state = batch_states[mini_batch_indices]
                _, n_logprobs, n_probs = agent.act(_state.view(_state.size(0), -1))
                entropy = n_probs.entropy()
                new_value = agent.critic(_state.view(_state.size(0), -1))

                # Calculate the policy loss
                ratio = get_ratio(n_logprobs, batch_logprobs[mini_batch_indices])
                policy_objective = get_policy_objective(mini_batch_advantages, ratio, clip_coeff=CLIP_COEF)
                policy_loss = -policy_objective

                # Calculate the value loss
                value_loss = get_value_loss(new_value.view(-1), batch_values[mini_batch_indices], batch_returns[mini_batch_indices], clip_coeff=CLIP_COEF)

                # Calculate the entropy loss
                entropy_objective = entropy.mean()

                # Combine losses to get the total loss
                total_loss = get_total_loss(policy_objective, value_loss, entropy_objective, value_loss_coeff=VALUE_LOSS_COEF, entropy_coeff=ENTROPY_COEF)

                optimizer.zero_grad()
                total_loss.backward()
                # Clip the gradient to stabilize training
                nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
                optimizer.step()

                total_actor_loss += policy_loss.item()
                total_critic_loss += value_loss.item()
                total_entropy_objective += entropy_objective.item()

        actor_loss_history.append(total_actor_loss // NUM_EPOCHS)
        critic_loss_history.append(total_critic_loss // NUM_EPOCHS)
        entropy_objective_history.append(total_entropy_objective // NUM_EPOCHS)

        # Prepare data for live plotting
        data_to_plot['Total Reward'].extend(reward_history)
        data_to_plot['Actor Loss'].extend(actor_loss_history)
        data_to_plot['Critic Loss'].extend(critic_loss_history) 
        data_to_plot['Entropy'].extend(entropy_objective_history)
    # Close the environment after training
    plot_set(data_to_plot)
    env.close()

if __name__ == "__main__":
    main()