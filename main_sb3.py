import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
import numpy as np
from gymnasium import spaces
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
import numpy as np
import os
from datetime import datetime
from gymnasium.envs.registration import register
# Add this after your imports
from gymnasium import spaces

class ActionSpaceWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        # Define proper action space for turn and accelerations
        self.action_space = spaces.Box(
            low=np.array([-10, -3, -3], dtype=np.float32),  # [turn, h_acc, v_acc]
            high=np.array([10, 3, 3], dtype=np.float32),    # Based on your max values
            dtype=np.float32
        )

    def step(self, action):
        # Convert the action to a dictionary format that your environment expects
        action_dict = {
            'turn': float(action[0]),
            'change_horizontal_acceleration': float(action[1]),
            'change_vertical_acceleration': float(action[2])
        }
        return self.env.step(action_dict)


# Import your environment
from src.atc_env_rendered import ATCplanning

# Register the environment with Gymnasium
register(
    id='ATCApproach-v1',  # Changed ID to avoid namespace issues
    entry_point='src.atc_env_rendered:ATCplanning',
    max_episode_steps=500,
)

# Add this right after the imports, before the FlattenEnv class:
def test_env():
    print("Testing environment...")
    env = ATCplanning()
    print("\nAction space:", env.action_space if hasattr(env, 'action_space') else "No action_space defined")
    obs, _ = env.reset()
    print("\nRaw observation:")
    print(obs)
    print("\nObservation structure:")
    for key, value in obs.items():
        print(f"\n{key}:")
        for k, v in value.items():
            print(f"  {k}: {type(v)}, shape: {np.shape(v) if hasattr(v, 'shape') else 'scalar'}, value: {v}")
    return obs

class FlattenEnv(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        # Define flattened observation space
        self.observation_space = spaces.Box(
            low=np.array([
                0, 0,           # agent position (2)
                0,             # agent heading (1)
                3000,          # agent altitude (1)
                -300, -300,    # agent speed (2)
                0, 0,          # target position (2)
                0,             # target heading (1)
                3000,          # target altitude (1)
                -300, -300     # target speed (2)
            ], dtype=np.float32),
            high=np.array([
                10000, 10000,  # agent position (2)
                360,           # agent heading (1)
                15000,         # agent altitude (1)
                300, 300,      # agent speed (2)
                10000, 10000,  # target position (2)
                360,           # target heading (1)
                15000,         # target altitude (1)
                300, 300       # target speed (2)
            ], dtype=np.float32),
            shape=(12,),
            dtype=np.float32
        )
        
        # Define action space
        self.action_space = spaces.Box(
            low=np.array([-10, -3, -3], dtype=np.float32),
            high=np.array([10, 3, 3], dtype=np.float32),
            dtype=np.float32
        )

    def _flatten_obs(self, obs):
        # Flatten observation dictionary into array
        flat_obs = np.array([
            *obs['agent']['position'],
            float(obs['agent']['heading'][0]),
            float(obs['agent']['altitude'][0]),
            *obs['agent']['speed'],
            *obs['target']['position'],
            float(obs['target']['heading'][0]),
            float(obs['target']['altitude'][0]),
            *obs['target']['speed']
        ], dtype=np.float32)
        return flat_obs

    def _unflatten_action(self, action):
        return {
            'turn': float(action[0]),
            'change_horizontal_acceleration': float(action[1]),
            'change_vertical_acceleration': float(action[2])
        }

    def step(self, action):
        action_dict = self._unflatten_action(action)
        obs, reward, terminated, truncated, info = self.env.step(action_dict)
        return self._flatten_obs(obs), reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._flatten_obs(obs), info

def make_env():
    env = ATCplanning()
    env = FlattenEnv(env)
    env = Monitor(env)
    return env

def train_model(total_timesteps=300_000, save_dir="models"):
    # Create and wrap the environment
    env = DummyVecEnv([make_env for _ in range(4)])
    env = VecNormalize(
        env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.,
        clip_reward=10.,
        gamma=0.99
    )

    # Create the model
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        policy_kwargs=dict(
            net_arch=dict(
                pi=[256, 256],
                vf=[256, 256]
            )
        ),
        tensorboard_log=save_dir
    )

    # Setup evaluation environment
    eval_env = DummyVecEnv([make_env])
    eval_env = VecNormalize(
        eval_env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.,
        clip_reward=10.,
        gamma=0.99
    )
    
    # Setup evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"{save_dir}/best_model",
        log_path=f"{save_dir}/eval_logs",
        eval_freq=10000,
        deterministic=True,
        render=False
    )

    try:
        # Train the model
        model.learn(
            total_timesteps=total_timesteps,
            callback=eval_callback,
            progress_bar=True
        )
        
        # Save the final model
        model.save(f"{save_dir}/final_model")
        env.save(f"{save_dir}/vec_normalize.pkl")
        
        return model, env
        
    except Exception as e:
        print(f"Training error: {e}")
        raise

if __name__ == "__main__":
    import os
    
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    # Train the model
    model, env = train_model()