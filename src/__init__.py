from gymnasium.envs.registration import register

register(
    id="env/Approach-v1",
    entry_point="src.atc_env_rendered:ATCplanning",
    max_episode_steps=500
)