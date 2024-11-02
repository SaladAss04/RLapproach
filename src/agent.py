from collections import defaultdict
import gymnasium as gym
import numpy as np

class DummyAgent:
    def __init__(self,
                env         
                ) -> None:
        self.env = env
    
    def get_action(self, obs: tuple[dict, dict]) -> dict:
        '''
        observation is the tuple of two dictionaries: agent_state and target_state,
            each: {'position', 'altitude', 'heading', 'speed'}
        action (output) is a dictionary: {'turn', 'horizontal_acceleration', 'vertical_acceleration'}
        '''
        agent, target = obs
        assert agent['position'].dim == 2 and agent['position'].dtype == np.float16
        assert target['position'].dim == 2 and target['position'].dtype == np.float16
        assert agent['speed'].dim == 2 and agent['speed'].dtype == np.float16
        assert target['speed'].dim == 2 and target['speed'].dtype == np.float16
        assert isinstance(agent['heading']) and 0 <= agent['heading'] <= 360
        assert isinstance(target['heading']) and 0 <= target['heading'] <= 360
        
        action =  {'turn': np.random.random_integers(0, 360, size=1, dtype=int),
                'h_acc': self.env.rng.uniform(-self.env.max_acc, -self.env.max_acc, size=1, dtype=np.float16),
                'v_acc': self.env.rng.uniform(-self.env.max_acc, -self.env.max_acc, size=1, dtype=np.float16)}
        
        acc = np.array(action['h_acc'], action['v_acc'], dtype=np.float16)
        norm_v = np.linalg.norm(acc, ord=2)
        if norm_v > self.max_acc:
            acc = acc * (self.max_acc / norm_v)
            action['h_acc'] = acc[0]
            action['v_acc'] = acc[1]
        
        return action
    
    def update(self):
        pass