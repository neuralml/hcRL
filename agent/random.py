import numpy as np


class RandomAgent:
    def __init__(
        self,
        name,
        num_obs,
        num_actions,
        hidden_size,
        device,
        lr,
        gamma,
        use_tb,
        use_wandb,
        batch_size,
        tau,
        nsteps,
        update_every_steps,
    ):
        self.num_obs = num_obs
        self.num_actions = num_actions
        self.hidden_size = hidden_size
        self.device = device
        self.lr = lr
        self.gamma = gamma
        self.use_tb = use_tb
        self.use_wandb = use_wandb
        self.batch_size = batch_size
        self.tau = tau
        self.nsteps = nsteps
        self.update_every_steps = update_every_steps

    def reset(self, replay_buffer=None, step=0, reward=None):
        pass

    def act(self, obs, step, eval_mode, eps=0.1):
        return np.random.randint(0, self.num_actions)

    def update(self, replay_buffer, step):
        metrics = dict()

        if step % self.update_every_steps != 0:
            return metrics

        return metrics
