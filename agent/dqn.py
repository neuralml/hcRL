import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4_allo = nn.Linear(hidden_size, output_size)
        self.fc4_ego = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x_allo = self.fc4_allo(x)
        x_ego = self.fc4_ego(x)
        return x_ego, x_allo


class DQNAgent:
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
        max_eps,
        min_eps,
        eps_decay,
    ):
        self.name = name
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
        self.max_eps = max_eps
        self.min_eps = min_eps
        self.eps_decay = eps_decay

        # extra params
        self.df_grad = None

        # models
        self.model = Net(num_obs, hidden_size, num_actions).to(device)
        self.target_model = Net(num_obs, hidden_size, num_actions).to(device)
        self.target_model.load_state_dict(self.model.state_dict())

        # optimizers
        self.model_opt = torch.optim.Adam(self.model.parameters(), lr=lr)

    def train(self, training=True):
        self.training = training
        self.model.train(training)

    def init_from(self, other):
        # copy parameters over
        self.model.load_state_dict(other["model_state_dict"])
        self.target_model.load_state_dict(other["model_state_dict"])
        print("Loading model state_dict")

    def get_epsilon(self):
        return self.eps

    def reset(self, replay_buffer=None, step=0, reward=None, update=False):
        self.eps = self.max_eps

    def add_penalty(self):
        return None

    def act(self, obs, step, eval_mode, eps=0.1, trial_info=None):
        if eps is not None:
            self.eps = eps

        if trial_info["Strategy"] in [3, 4]:
            strategy = "ego"
        else:
            strategy = "allo"

        obs = torch.FloatTensor(obs).to(device=self.device).unsqueeze(0)
        if np.random.random() >= self.get_epsilon():
            q_val_ego, q_val_allo = self.model(obs)
            if strategy == "ego":
                q_val = q_val_ego
            else:
                q_val = q_val_allo
            action = torch.max(q_val, 1)[1]

            return action.data.cpu().numpy()[0], q_val, "model"
        else:
            return np.random.randint(0, self.num_actions), None, "random"

    def update_critic(
        self,
        state,
        action,
        reward,
        next_state,
        step,
        trial_info=None,
    ):
        metrics = dict()

        if trial_info["Strategy"] in [3, 4]:
            strategy = "ego"
        else:
            strategy = "allo"

        current_q_values_ego, current_q_values_allo = self.model(state)
        next_q_val_ego, next_q_val_allo = self.target_model(next_state)
        if strategy == "ego":
            current_q_values = current_q_values_ego
            next_q_val = next_q_val_ego
        else:
            current_q_values = current_q_values_allo
            next_q_val = next_q_val_allo

        current_q_values = current_q_values.gather(1, action)
        next_q_val = next_q_val.detach().max(1)[0]

        new_q = (next_q_val * self.gamma) + reward

        loss = F.mse_loss(current_q_values, new_q)

        penalty = self.add_penalty()
        if penalty is not None:
            loss = loss + penalty

        if self.use_tb or self.use_wandb:
            metrics["loss"] = loss.item()
            metrics["current_q_val"] = current_q_values.mean().item()
            metrics["target_q"] = new_q.mean().item()
            if penalty is not None:
                metrics["penalty"] = penalty.item()

        self.model_opt.zero_grad(set_to_none=True)
        loss.backward()
        self.model_opt.step()

        # update target model
        self.soft_update_params(self.model, self.target_model, self.tau)

        return metrics

    def soft_update_params(self, net, target_net, tau):
        for param, target_param in zip(net.parameters(), target_net.parameters()):
            target_param.data.lerp_(param.data, tau)

    def update(self, replay_buffer, step, trial_info=None):
        metrics = dict()

        if step % self.update_every_steps != 0:
            return metrics

        if len(replay_buffer) < replay_buffer.capacity:
            return metrics

        state, action, reward, next_state = replay_buffer.sample(self.batch_size)

        if self.use_tb or self.use_wandb:
            metrics["batch_reward"] = reward.mean().item()

        metrics.update(
            self.update_critic(
                state,
                action,
                reward,
                next_state,
                step,
                trial_info=trial_info,
            )
        )

        return metrics
