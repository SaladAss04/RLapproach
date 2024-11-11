from src.agent.agent import PPOModel
import gymnasium as gym
from src.components import *
from src.components import NUM_ENVS
from src.env.environment import DummyEnv
from src.env.atc_env import ATCplanning
from src.evaluation import evaluate_policy
import torch
from tqdm import tqdm

NUM_ITERATIONS = 3000
EVAL_FREQUENCY = 100
def make_env():
    return gym.make('env/Approach-v1')

def create_parallel_envs(num = NUM_ENVS):
    ret = gym.vector.SyncVectorEnv([make_env for _ in range(num)])
    return ret

def main():
    '''
    We use multiple parallel envs to train at the same time.
    Train for multiple iterations, during each iteration one 'rollout-training' is performed.
    '''
    envs = create_parallel_envs() 
    eval_env = make_env()
    agent = PPOModel()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # track success
    success_rate_window = 100
    success_history = []
    best_eval_reward = float('-inf')

    # lr scheduling
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        agent.actor_optimizer, mode='max', factor=0.5, patience=10,
        verbose=True, min_lr=1e-6
    )
    
    legend = {
        "reward":[],
        "steps":[],
        "actor_loss":[],
        "critic_loss":[],
        "entropy_objective":[],
        "eval_reward":[],
        "success_rate":[]
    }

    best_eval_reward = float('-inf')

    
    for iter in tqdm(range(NUM_ITERATIONS)):
        render = False
        if iter % 100 == 0:
            render = True

        states, actions, logprobs, rewards, dones, values, r_h, e_h = rollout(agent, envs, device, render)
        advantage = returns(agent, states, actions, rewards, dones, values, device)
        a_loss, c_loss, e_obj = train(agent, envs, states, actions, logprobs, values, advantage)
        legend['reward'].extend(r_h)
        legend['steps'].extend(e_h)
        legend['actor_loss'].append(a_loss)
        legend['critic_loss'].append(c_loss)
        legend['entropy_objective'].append(e_obj)

        if iter % EVAL_FREQUENCY == 0:
            eval_reward, success_rate = evaluate_policy(agent, eval_env, device, num_episodes=20)
            success_history.append(success_rate)
            scheduler.step(eval_reward)

            if eval_reward > best_eval_reward:
                best_eval_reward = eval_reward
                torch.save({
                               'model_state_dict': agent.state_dict(),
                               'optimizer_state_dict': agent.actor_optimizer.state_dict(),
                               'iteration': iter,
                               'eval_reward': eval_reward
                           }, 'best_model.pth')

            legend['eval_reward'].append(eval_reward)
            legend['success_rate'].append(success_rate)
            
            print(f"\nIteration {iter}")
            print(f"Training - average reward: {np.mean(r_h):.2f}")
            print(f"Evaluation - average reward: {eval_reward:.2f}")
            print(f"Success rate: {success_rate:.2%}")
            print(f"Actor loss: {a_loss:.2f}")
            print(f"Critic loss: {c_loss:.2f}")
            print(f"Entropy: {e_obj:.2f}")

        if len(success_history) > 5 and np.mean(success_history[-5:]) > 0.95:
            print("Success rate threshold reached. Stopping training.")
            break

        
    plot_set(legend) 
    envs.close()
    eval_env.close()
    torch.save(agent.state_dict(), 'final_model.pth')
    
if __name__ == "__main__":
    main()
