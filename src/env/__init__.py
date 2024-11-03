from gymnasium.envs.registration import register

register(
    id="env/Approach-v0",
    entry_point="src.env.environment:DummyEnv",
    max_episode_steps=500
)