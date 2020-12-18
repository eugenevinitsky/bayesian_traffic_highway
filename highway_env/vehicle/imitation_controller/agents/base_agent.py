class BaseAgent:
    def __init__(self, env, agent_params):
        super(BaseAgent, self).__init__(**kwargs)

    def train(self, ob_no, ac_na, re_n, next_ob_no, terminal_n):
        """Return a dictionary of logging information."""
        raise NotImplementedError

    def add_to_replay_buffer(self, paths):
        raise NotImplementedError

    def sample(self, batch_size):
        raise NotImplementedError

    def save(self, path):
        raise NotImplementedError
