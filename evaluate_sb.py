import gymnasium as gym
from atc_env_rendered import ATCplanning
from wrapper import *
from stable_baselines3 import PPO


env = make_env()
model = PPO.load(path = 'best_model.zip', env = env)

reward_val = 0
while reward_val < 1:
    obs = env.reset()  # 重置环境以获得初始观测
    done = False
    obs = obs[0]
    while not done:
        # 使用模型预测动作
        action, _states = model.predict(obs, deterministic=True)  # 设置 deterministic=True 确保输出最优动作

        # 执行动作并获得新的观测、奖励等
        obs, reward, done, truncated, info = env.step(action)
        if reward > 0:
            print(reward)
        done = done or truncated
        # 渲染环境（可选）
        env.render()

    env.close()


