from gymnasium.envs.registration import register

register(
    id="env/Approach-v1",
    entry_point="src.env.atc_env_rendered:ATCplanning",
    max_episode_steps=500
)

register(
    id="env/Approach-v2",
    entry_point="src.env.atc_env_rendered_sb3:ATCplanning",
    max_episode_steps=500
)
