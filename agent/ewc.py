from copy import deepcopy

import torch
import torch.nn.functional as F

from agent.dqn import DQNAgent


class EWCAgent(DQNAgent):
    def __init__(self, ewc_importance, **kwargs):
        super().__init__(**kwargs)

        self.ewc_importance = ewc_importance
        self.precision_matrices = None

        self.params = {
            n: p for n, p in self.model.named_parameters() if p.requires_grad
        }
        self._means = {}

        for n, p in deepcopy(self.params).items():
            self._means[n] = p.data

    def _diag_fisher(self, dataset, trial_info=None):
        print("updating diag_fish")
        precision_matrices = {}
        for n, p in deepcopy(self.params).items():
            p.data.zero_()
            precision_matrices[n] = p.data

        self.model.eval()

        if trial_info["Strategy"] in [3, 4]:
            strategy = "ego"
        else:
            strategy = "allo"

        for transition in dataset.memory:
            state = torch.as_tensor(transition.state, device=self.device).float()
            action = torch.as_tensor(transition.action, device=self.device).unsqueeze(
                -1
            )
            reward = torch.as_tensor(transition.reward, device=self.device).unsqueeze(
                -1
            )
            next_state = torch.as_tensor(
                transition.next_state, device=self.device
            ).float()

            state = state.unsqueeze(0)
            next_state = next_state.unsqueeze(0)
            action = action.unsqueeze(-1)
            reward = reward.unsqueeze(-1)

            # if train mode
            self.model.zero_grad()

            # # 1-head
            # output = self.model(state)
            # pred = self.target_model(next_state)

            # 2-head
            current_q_values_ego, current_q_values_allo = self.model(state)
            next_q_val_ego, next_q_val_allo = self.target_model(next_state)
            if strategy == "ego":
                output = current_q_values_ego
                pred = next_q_val_ego
            else:
                output = current_q_values_allo
                pred = next_q_val_allo

            output = output.gather(1, action)
            pred = pred.detach().max(1)[0]

            new_q = (pred * self.gamma) + reward

            loss = F.mse_loss(output, new_q)
            loss.backward()

            for n, p in self.model.named_parameters():
                if p.grad is not None:
                    precision_matrices[n].data += p.grad.data**2 / len(dataset)

            # # else:
            # # if in test mode
            # with torch.no_grad():
            #     # 1-head
            #     output = self.model(state)
            #     pred = self.target_model(next_state)
            #     output = output.gather(1, action)
            #     pred = pred.detach().max(1)[0]
            #     new_q = (pred * self.gamma) + reward
            #     loss = F.mse_loss(output, new_q)
            #     # for n, p in self.model.named_parameters():
            #     #     precision_matrices[n].data += p.grad.data**2 / len(dataset)

        # precision_matrices = {n: p for n, p in precision_matrices.items()}
        return precision_matrices

    def penalty(self):
        loss = 0
        if self.precision_matrices is None:
            return loss
        for n, p in self.model.named_parameters():
            _loss = self.precision_matrices[n] * (p - self._means[n]) ** 2
            loss += _loss.sum()

        return loss

    def reset(self, replay_buffer=None, step=None, reward=None, update=False):
        self.precision_matrices = self._diag_fisher(replay_buffer, trial_info=step)

    def add_penalty(self):
        return self.ewc_importance * self.penalty()
