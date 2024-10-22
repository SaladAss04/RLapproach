from gym.envs.registration import register

register(
    id='AtcEnv-v0',
    entry_point='envs.atc_env:ATCEnv',
)