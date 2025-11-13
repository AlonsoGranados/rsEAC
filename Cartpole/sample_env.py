import numpy as np
class EnvSampler():
    def __init__(self, env, max_path_length=1000):
        self.env = env

        self.path_length = 0
        self.current_state = None
        self.max_path_length = max_path_length
        self.action = None

    def sample(self, agent, eval_t=False):
        if self.current_state is None:
            self.current_state, _ = self.env.reset()

        cur_state = self.current_state

        action = agent.select_action(self.current_state, eval_t)

        # if self.path_length % 4 == 0:
        #     action = agent.select_action(self.current_state, eval_t)
        #     self.action = action
        # else:
        #     action = self.action

        next_state, reward, terminated, _, info = self.env.step(action)


        # reward = np.clip(reward, a_min = -10, a_max=10)
        # print(reward)
        # if next_state[0] > 0:
        #     reward = reward + np.random.randn()

        self.path_length += 1

        if terminated or self.path_length >= self.max_path_length:
            self.current_state = None
            self.path_length = 0
            self.action = None
        else:
            self.current_state = next_state

        return cur_state, action, next_state, reward, terminated, info

    def exploratory_sample(self):
        if self.current_state is None:
            self.current_state, _ = self.env.reset()

        cur_state = self.current_state
        action = self.env.action_space.sample()

        next_state, reward, terminated, _, info = self.env.step(action)
        # print(reward)
        reward = np.clip(reward, a_min = -20, a_max=20)
        # print(reward)

        # if next_state[0] > 0:
        #     reward = reward + np.random.randn()

        self.path_length += 1

        if terminated or self.path_length >= self.max_path_length:
            self.current_state = None
            self.path_length = 0
        else:
            self.current_state = next_state

        return cur_state, action, next_state, reward, terminated, info