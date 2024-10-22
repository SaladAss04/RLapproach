def render(self, mode='human'):
    """Render the current environment state."""
    if self.viewer is None:
        from .rendering import ATCRenderer
        self.viewer = ATCRenderer()
    
    # Prepare state for renderer
    env_state = {
        'config': self.config,
        'aircraft': self.aircraft,
        'steps': self.steps,
        'reward': self.last_reward if hasattr(self, 'last_reward') else 0
    }
    
    return self.viewer.render(env_state, mode)

def close(self):
    """Clean up environment resources."""
    if self.viewer:
        self.viewer.close()
        self.viewer = None