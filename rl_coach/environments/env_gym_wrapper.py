import numpy as np
import gym

from rl_coach.environments.control_suite_environment import ControlSuiteEnvironment
from rl_coach.environments.control_suite_environment import ControlSuiteEnvironmentParameters, control_suite_envs
from rl_coach.environments.environment import SingleLevelSelection
from rl_coach.base_parameters import VisualizationParameters

class EnvGymWrapper():
    def __init__(self, env_id, seed=None):
        self.env_id = env_id
        self.env_name = env_id.split(':')[0]

        env_params = ControlSuiteEnvironmentParameters(level=env_id)
        vis_params = VisualizationParameters(render=True)

        if seed is not None:
            env_params.set_seed(seed)

        self.env = ControlSuiteEnvironment(**env_params.__dict__, visualization_parameters=vis_params)

        self.state_space = self.env.state_space
        self.observation_space = gym.spaces.box.Box(low=self.env.state_space['measurements'].low, high=self.env.state_space['measurements'].high)
        self.action_space = gym.spaces.box.Box(low=self.env.action_space.low, high=self.env.action_space.high)
        self.reward_range = (-np.inf, np.inf)
        self.metadata = {}

        gym_env_names = gym.envs.registry.env_specs.keys()
        gym_env_name = get_similar_env(self.env_name, gym_env_names)
        self.spec = gym.envs.registry.env_specs[gym_env_name]
        return

    def seed(self, seed):
        # TODO
        return

    def reset(self):
        self.env._restart_environment_episode()
        return self.env.measurements

    #new_obs, rewards, dones, infos = env.step(clipped_actions)
    def step(self, action):
        last_env_response = self.env.step(action)

        #new_vision = last_env_response._next_state['pixels']
        new_obs = last_env_response._next_state['measurements']
        rewards = last_env_response._reward
        dones = last_env_response._game_over
        infos = last_env_response.info

        return new_obs, rewards, dones, infos

    def close(self):
        self.env.close()
        return

def get_similar_env(query_env_name, env_names):
    for env_name in env_names:
        if query_env_name == env_name:
            return env_name

    for env_name in env_names:
        if query_env_name.lower() == env_name.split('-')[0].lower():
            return env_name

    for env_name in env_names:
        if query_env_name.lower() in env_name.lower():
            return env_name

    return None