import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils


class DRQN(nn.Module):
    def __init__(
        self,
        input_shape,
        num_actions,
        device,
        gru_size=200,
        hidden_size=200,
        bidirectional=False,
    ):
        super(DRQN, self).__init__()

        self.input_shape = input_shape
        self.num_actions = num_actions
        self.gru_size = gru_size
        self.hidden = hidden_size
        self.device = device
        self.bidirectional = bidirectional
        self.num_directions = 1  # 2 if self.bidirectional else 1

        self.in_fc1 = nn.Linear(self.input_shape, self.gru_size)
        self.gru = nn.GRU(
            self.gru_size,
            self.gru_size,
            num_layers=1,
            batch_first=True,
            bidirectional=bidirectional,
        )
        self.fc1 = nn.Linear(self.gru_size, self.hidden)
        self.fc2_allo = nn.Linear(self.hidden, self.num_actions)
        self.fc2_ego = nn.Linear(self.hidden, self.num_actions)

    def forward(self, x, hx=None, info="train"):
        batch_size = x.size(0)
        sequence_length = x.size(1)

        x = F.relu(self.in_fc1(x))
        x = x.view((-1, self.gru_size))

        # format outp for batch first gru
        feats = x.view(batch_size, sequence_length, -1)
        hidden = self.init_hidden(batch_size) if hx is None else hx
        out, hidden = self.gru(feats, hidden)

        x = F.relu(self.fc1(out))
        x_ego = self.fc2_ego(x)
        x_allo = self.fc2_allo(x)

        return x_ego, x_allo, hidden, x

    def init_hidden(self, batch_size):
        return torch.zeros(
            1 * self.num_directions,
            batch_size,
            self.gru_size,
            device=self.device,
            dtype=torch.float,
        )

    def sample_noise(self):
        pass


class DRQNAgent:
    def __init__(
        self,
        name,
        num_obs,
        num_actions,
        neurons_rnn_layer,
        hidden_size,
        device,
        lr,
        gamma,
        sequence_length,
        use_tb,
        use_wandb,
        batch_size,
        tau,
        nsteps,
        update_every_steps,
        max_eps,
        min_eps,
        eps_decay,
        n_heads,
    ):
        self.name = name
        self.num_obs = num_obs
        self.num_actions = num_actions
        self.neurons_rnn_layer = neurons_rnn_layer
        self.hidden_size = hidden_size
        self.device = device
        self.lr = lr
        self.gamma = gamma
        self.sequence_length = sequence_length
        self.use_tb = use_tb
        self.use_wandb = use_wandb
        self.batch_size = batch_size
        self.tau = tau
        self.nsteps = nsteps
        self.update_every_steps = update_every_steps
        self.max_eps = max_eps
        self.min_eps = min_eps
        self.eps_decay = eps_decay
        self.n_heads = n_heads

        # models
        self.model = DRQN(
            num_obs, num_actions, device, neurons_rnn_layer, hidden_size
        ).to(device)
        self.target_model = DRQN(
            num_obs, num_actions, device, neurons_rnn_layer, hidden_size
        ).to(device)
        self.target_model.load_state_dict(self.model.state_dict())

        # optimizers
        self.model_opt = torch.optim.Adam(self.model.parameters(), lr=lr)

        self.reset_hx()

    def train(self, training=True):
        self.training = training
        self.model.train(training)

    def init_from(self, other):
        # copy parameters over
        self.model.load_state_dict(other["model_state_dict"])
        self.target_model.load_state_dict(other["model_state_dict"])
        print("Loading model state_dict")

    def get_epsilon(self, strategy):
        return self.eps

    def reset(self, replay_buffer=None, step=0, reward=False, update=False):
        self.reset_hx()
        # pass

    def huber(self, x):
        cond = (x.abs() < 1.0).float().detach()
        return 0.5 * x.pow(2) * cond + (x.abs() - 0.5) * (1.0 - cond)

    def act(self, obs, step, eval_mode, eps=None, trial_info=None):
        if eps is not None:
            self.eps = eps

        if trial_info["Strategy"] in [3, 4]:
            strategy = "ego"
        else:
            strategy = "allo"

        with torch.no_grad():
            self.seq.pop(0)
            self.seq.append(obs)
            if np.random.random() >= self.get_epsilon(strategy):
                X = torch.tensor(
                    np.array([self.seq]), device=self.device, dtype=torch.float
                )
                self.model.sample_noise()
                a_ego, a_allo, _, _ = self.model(X)
                if strategy == "ego":
                    a = a_ego
                else:
                    a = a_allo
                val = a[:, -1, :]  # select last element of seq
                a = val.max(1)[1]

                return a.item(), val, "model"
            else:
                return np.random.randint(0, self.num_actions), None, "random"

    def reset_hx(self):
        self.seq = [np.zeros(self.num_obs) for j in range(self.sequence_length)]

    def update_critic(
        self,
        state,
        action,
        reward,
        next_state,
        non_final_mask,
        empty_next_state_values,
        step,
        trial_info,
    ):
        metrics = dict()
        if trial_info["Strategy"] in [3, 4]:
            strategy = "ego"
        else:
            strategy = "allo"

        curr_q_val_ego, curr_q_val_allo, _, _ = self.model(state)
        if strategy == "ego":
            current_q_values = curr_q_val_ego
        else:
            current_q_values = curr_q_val_allo
        current_q_values = current_q_values.gather(2, action).squeeze()

        with torch.no_grad():
            max_next_q_values = torch.zeros(
                (self.batch_size, self.sequence_length),
                device=self.device,
                dtype=torch.float,
            )
            if not empty_next_state_values:
                max_next_ego, max_next_allo, _, _ = self.target_model(next_state)
                if strategy == "ego":
                    max_next = max_next_ego
                else:
                    max_next = max_next_allo
                max_next_q_values[non_final_mask] = max_next.max(dim=2)[0]
            expected_q_values = reward + (
                (self.gamma**self.nsteps) * max_next_q_values
            )

        diff = expected_q_values - current_q_values
        loss = self.huber(diff)
        loss = loss.mean()

        if self.use_tb or self.use_wandb:
            metrics["loss"] = loss.item()
            metrics["current_q_val"] = current_q_values.mean().item()
            metrics["expected_q_val"] = expected_q_values.mean().item()

        self.model_opt.zero_grad(set_to_none=True)
        loss.backward()
        for param in self.model.parameters():
            if param.grad is not None:
                param.grad.data.clamp_(-1, 1)
        self.model_opt.step()

        # update target model
        utils.soft_update_params(self.model, self.target_model, self.tau)

        return metrics

    def update(self, replay_buffer, step, trial_info):
        metrics = dict()

        if step % self.update_every_steps != 0:
            return metrics

        if len(replay_buffer) < 2:
            return metrics

        batch, indices, weights = replay_buffer.sample(self.batch_size)
        state, action, reward, next_state = zip(*batch)
        shape = (self.batch_size, self.sequence_length) + (self.num_obs,)
        state = torch.tensor(
            np.array(state), device=self.device, dtype=torch.float
        ).view(shape)
        action = torch.tensor(action, device=self.device, dtype=torch.long).view(
            self.batch_size, self.sequence_length, -1
        )
        reward = torch.tensor(reward, device=self.device, dtype=torch.float).view(
            self.batch_size, self.sequence_length
        )
        next_state = tuple(
            [
                next_state[i]
                for i in range(len(next_state))
                if (i + 1) % (self.sequence_length) == 0
            ]
        )
        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, next_state)),
            device=self.device,
            dtype=torch.bool,
        )  # uint8

        try:  # sometimes all next states are false, especially with nstep returns
            non_final_next_states = torch.tensor(
                np.array([s for s in next_state if s is not None]),
                device=self.device,
                dtype=torch.float,
            ).unsqueeze(dim=1)
            non_final_next_states = torch.cat(
                [state[non_final_mask, 1:, :], non_final_next_states], dim=1
            )
            empty_next_state_values = False
        except:
            empty_next_state_values = True

        if self.use_tb or self.use_wandb:
            metrics["batch_reward"] = reward.mean().item()

        metrics.update(
            self.update_critic(
                state,
                action,
                reward,
                non_final_next_states,
                non_final_mask,
                empty_next_state_values,
                step,
                trial_info,
            )
        )

        return metrics
