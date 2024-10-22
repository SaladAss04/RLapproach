from gym.envs.registration import register

# Register the environment
register(
    id='AtcEnv-v0',
    entry_point='envs.atc_env:ATCEnv',
)

# Make the environment class available when importing envs
from .atc_env import ATCEnv