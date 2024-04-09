import csv
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F


def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


def hard_update_params(net, target_net):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(param.data)


def to_torch(xs, device):
    return tuple(torch.as_tensor(x, device=device) for x in xs)


class Timer:
    def __init__(self):
        self._start_time = time.time()
        self._last_time = time.time()

    def reset(self):
        elapsed_time = time.time() - self._last_time
        self._last_time = time.time()
        total_time = time.time() - self._start_time
        return elapsed_time, total_time

    def total_time(self):
        return time.time() - self._start_time


class PerformanceStats(object):

    """Docstring for Heatmaps."""

    def __init__(self, root_dir, csv_read):
        """TODO: to be defined."""
        if root_dir is not None:
            self.save_dir = root_dir / "heatmaps_and_stats"
            self.save_dir.mkdir(exist_ok=True)
        else:
            self.save_dir = None
        self.trial_counter = {}
        self.env_configuration = {}
        # the env_configuration will look like a dict ->
        # {'strategy_val': {'start_val': [9x9 matrix]}}
        self.allo = np.zeros(np.max(csv_read["Trial"]))
        self.ego = np.zeros(np.max(csv_read["Trial"]))
        self.env_actions = []
        self.values = []

    def record_position(self, position, trial):
        strategy = trial["Strategy"]
        start_zone = trial["Start zone"]
        x, y = position[1], position[0]

        if strategy not in self.env_configuration:
            self.env_configuration[strategy] = {}
        if start_zone not in self.env_configuration[strategy]:
            self.env_configuration[strategy][start_zone] = np.zeros((9, 9))

        self.env_configuration[strategy][start_zone][x, y] += 1

    def record_actions(self, action_info, trial, timestep):
        """record if the action taken is random or coming out of the model"""
        strategy = trial["Strategy"]
        start_zone = trial["Start zone"]
        trial = trial["Trial"]
        action_step = {
            "Trial": trial,
            "Strategy": strategy,
            "Start zone": start_zone,
            "Timestep": timestep,
            "Action": action_info,
        }
        self.env_actions.append(action_step)

    def record_values(self, position, value, trial, timestep):
        strategy = trial["Strategy"]
        start_zone = trial["Start zone"]
        trial = trial["Trial"]
        # print(value)
        value_step = {
            "Trial": trial,
            "Strategy": strategy,
            "Start zone": start_zone,
            "Timestep": timestep,
            "Position": position,
            "Value_0": value[0],
            "Value_1": value[1],
            "Value_2": value[2],
        }

        self.values.append(value_step)

    def record_outcome(self, info, trial, env):
        strategy = trial["Strategy"]
        start_zone = trial["Start zone"]
        end_goal = (info["old_info"][0], info["old_info"][1])
        TrialEnv = env

        # update trial count
        if strategy not in self.trial_counter:
            self.trial_counter[strategy] = {}
        if start_zone not in self.trial_counter[strategy]:
            self.trial_counter[strategy][start_zone] = 0

        self.trial_counter[strategy][start_zone] += 1

        # record ego/allo outcomes
        if strategy == 1 and start_zone in [1] and end_goal == TrialEnv.EAST:
            self.allo[trial["Trial"] - 1] = 1
        elif strategy == 1 and start_zone in [2] and end_goal == TrialEnv.WEST:
            self.allo[trial["Trial"] - 1] = 1
        elif strategy == 1 and start_zone in [1] and end_goal == TrialEnv.WEST:
            self.allo[trial["Trial"] - 1] = 1
        elif strategy == 1 and start_zone in [2] and end_goal == TrialEnv.EAST:
            self.allo[trial["Trial"] - 1] = 1
        elif strategy == 2 and start_zone in [1] and end_goal == TrialEnv.EAST:
            self.allo[trial["Trial"] - 1] = 1
        elif strategy == 2 and start_zone in [2] and end_goal == TrialEnv.WEST:
            self.allo[trial["Trial"] - 1] = 1
            # elif strategy == 2 and start_zone in [1] and end_goal == TrialEnv.WEST:
            #     self.allo[trial["Trial"] - 1] = 1
            # elif strategy == 2 and start_zone in [2] and end_goal == TrialEnv.EAST:
            self.allo[trial["Trial"] - 1] = 1
        elif strategy == 3 and start_zone in [1, 2] and end_goal == TrialEnv.WEST:
            self.ego[trial["Trial"] - 1] = 1
        # elif strategy == 3 and start_zone == 2 and end_goal == TrialEnv.EAST:
        #     self.ego[trial["Trial"] - 1] = 1
        elif strategy == 4 and start_zone == 1 and end_goal == TrialEnv.EAST:
            self.ego[trial["Trial"] - 1] = 1
        elif strategy == 4 and start_zone == 2 and end_goal == TrialEnv.WEST:
            self.ego[trial["Trial"] - 1] = 1
        else:
            print("None of the conditions were met")

    def save_ego_allo(self):
        np.save(f"{self.save_dir}/DQN_allo", np.array(self.allo))
        np.save(f"{self.save_dir}/DQN_ego", np.array(self.ego))

    def save_actions(self):

        with open(f"{self.save_dir}/actions.csv", "w", newline="") as output_file:
            keys = self.env_actions[0].keys()
            dict_writer = csv.DictWriter(output_file, keys)
            dict_writer.writeheader()
            dict_writer.writerows(self.env_actions)
            print("Done saving actions")

    def save_values(self):
        with open(f"{self.save_dir}/values.csv", "w", newline="") as output_file:
            keys = self.values[0].keys()
            dict_writer = csv.DictWriter(output_file, keys)
            dict_writer.writeheader()
            dict_writer.writerows(self.values)
            print("Done saving values")

    def save_and_plot_heatmaps(self, trial_num):
        for strategy in self.env_configuration:
            for start_zone in self.env_configuration[strategy]:
                fig = plt.figure(figsize=(4, 4))
                ax = fig.add_subplot(1, 1, 1)
                mean_values = (
                    self.env_configuration[strategy][start_zone]
                    / self.trial_counter[strategy][start_zone]
                )
                sns.heatmap(mean_values, cmap="hot_r", ax=ax, annot=True)
                plt.savefig(
                    f"{self.save_dir}/heatmap_{strategy}_{start_zone}_{trial_num}.png",
                    dpi=400,
                )


class Activations(object):

    """Class to record and save the Activations."""

    def __init__(self, root_dir, agent):
        """TODO: to be defined."""
        if root_dir is not None:
            self.save_dir = root_dir / "activations"
            self.save_dir.mkdir(exist_ok=True)
        else:
            self.save_dir = None

        self.agent = agent
        self.activations_CA1 = []
        self.activation_track_CA1 = []
        self.activations_CA3 = []
        self.activation_track_CA3 = []
        self.activations_CA1_ego = []
        self.activation_track_CA1_ego = []
        self.activations_CA1_allo = []
        self.activation_track_CA1_allo = []
        if agent is not None:
            if agent.name == "drqn":
                self.agent.model.fc1.register_forward_hook(
                    self.get_DRQN_CA1_activation()
                )
                # self.agent.model.gru.register_forward_hook(
                #     self.get_DRQN_CA3_activation()
                # )
                # self.agent.model.fc1_ego.register_forward_hook(
                #     self.get_DRQN_CA1_ego_activation()
                # )
                # self.agent.model.fc1_allo.register_forward_hook(
                #     self.get_DRQN_CA1_allo_activation()
                # )
            else:
                self.agent.model.fc3.register_forward_hook(
                    self.get_DQN_CA1_activation()
                )
                # self.agent.model.fc3_ego.register_forward_hook(
                #     self.get_DQN_CA1_ego_activation()
                # )
                # self.agent.model.fc3_allo.register_forward_hook(
                #     self.get_DQN_CA1_allo_activation()
                # )

    def get_DRQN_CA1_activation(self):
        def hook(model, input, output):
            if len(output) == 1:
                self.activations_CA1.append(F.relu(output[0][0].detach()))

        return hook

    def get_DRQN_CA3_activation(self):
        def hook(model, input, output):
            if len(output[0]) == 1:
                self.activations_CA3.append(output[0][0].detach()[0])

        return hook

    def get_DRQN_CA1_ego_activation(self):
        def hook(model, input, output):
            if len(output) == 1:
                self.activations_CA1_ego.append(F.relu(output[0][0].detach()))

        return hook

    def get_DRQN_CA1_allo_activation(self):
        def hook(model, input, output):
            if len(output) == 1:
                self.activations_CA1_allo.append(F.relu(output[0][0].detach()))

        return hook

    def get_DQN_CA1_activation(self):
        def hook(model, input, output):
            if len(output) == 1:
                self.activations_CA1.append(F.relu(output.detach()[0]))

        return hook

    def get_DQN_CA1_ego_activation(self):
        def hook(model, input, output):
            if len(output) == 1:
                self.activations_CA1_ego.append(F.relu(output.detach()[0]))

        return hook

    def get_DQN_CA1_allo_activation(self):
        def hook(model, input, output):
            if len(output) == 1:
                self.activations_CA1_allo.append(F.relu(output.detach()[0]))

        return hook

    # def get_DQN_CA1_activation(self):
    #     def hook(model, input, output):
    #         if len(output) == 1:
    #             self.activations_CA1.append(F.relu(output.detach()[0]))

    #     return hook

    def add_activation(self, pos, trial):
        if self.agent is None:
            return

        for timestep, activation in enumerate(self.activations_CA1):
            for ind, neuron in enumerate(activation):
                track = {
                    "Trial": trial["Trial"],
                    "Strategy": trial["Strategy"],
                    "Timestep": timestep,
                    "Position": pos,
                    "Neuron": ind + 1,
                    "Activation": neuron.item(),
                }
                self.activation_track_CA1.append(track)

        # for timestep, activation in enumerate(self.activations_CA3):
        #     for ind, neuron in enumerate(activation):
        #         track = {
        #             "Trial": trial["Trial"],
        #             "Strategy": trial["Strategy"],
        #             "Timestep": timestep,
        #             "Position": pos,
        #             "Neuron": ind + 1,
        #             "Activation": neuron.item(),
        #         }
        #         self.activation_track_CA3.append(track)

        # for timestep, activation in enumerate(self.activations_CA1_ego):
        #     for ind, neuron in enumerate(activation):
        #         track = {
        #             "Trial": trial["Trial"],
        #             "Strategy": trial["Strategy"],
        #             "Timestep": timestep,
        #             "Neuron": ind + 1,
        #             "Activation": neuron.item(),
        #         }
        #         self.activation_track_CA1_ego.append(track)

        # for timestep, activation in enumerate(self.activations_CA1_allo):
        #     for ind, neuron in enumerate(activation):
        #         track = {
        #             "Trial": trial["Trial"],
        #             "Strategy": trial["Strategy"],
        #             "Timestep": timestep,
        #             "Neuron": ind + 1,
        #             "Activation": neuron.item(),
        #         }
        #         self.activation_track_CA1_allo.append(track)

        # self.reset()

    def reset(self):
        self.activations_CA1 = []
        self.activations_CA3 = []
        self.activations_CA1_ego = []
        self.activations_CA1_allo = []
        # print("Reset Activations")

    # def return_all_activations(self):
    #     return self.activations_track_CA1

    def save(self):
        if self.agent is None:
            return

        with open(
            f"{self.save_dir}/activations_CA1.csv", "w", newline=""
        ) as output_file:
            keys = self.activation_track_CA1[0].keys()
            dict_writer = csv.DictWriter(output_file, keys)
            dict_writer.writeheader()
            dict_writer.writerows(self.activation_track_CA1)
            print("Done saving CA1 Activations")

        # with open(
        #     f"{self.save_dir}/activations_CA3.csv", "w", newline=""
        # ) as output_file:
        #     keys = self.activation_track_CA3[0].keys()
        #     dict_writer = csv.DictWriter(output_file, keys)
        #     dict_writer.writeheader()
        #     dict_writer.writerows(self.activation_track_CA3)
        #     print("Done saving CA3 Activations")

        # with open(
        #     f"{self.save_dir}/activations_CA1_ego.csv", "w", newline=""
        # ) as output_file:
        #     keys = self.activation_track_CA1_ego[0].keys()
        #     dict_writer = csv.DictWriter(output_file, keys)
        #     dict_writer.writeheader()
        #     dict_writer.writerows(self.activation_track_CA1_ego)
        #     print("Done saving CA1 ego Activations")

        # with open(
        #     f"{self.save_dir}/activations_CA1_allo.csv", "w", newline=""
        # ) as output_file:
        #     keys = self.activation_track_CA1_allo[0].keys()
        #     dict_writer = csv.DictWriter(output_file, keys)
        #     dict_writer.writeheader()
        #     dict_writer.writerows(self.activation_track_CA1_allo)
        #     print("Done saving CA1 allo Activations")


def load_trial_data(num_trials=1000, data_source=None, csv_path=None):
    if data_source == "animal":
        # load animal data
        trials = list(pd.read_csv(csv_path).T.to_dict().values())
        csv_read = pd.read_csv(csv_path)
        eval_trials = []
        for i in range(1, 5):
            eval_trials.append({"Trial": 1, "Strategy": i, "Start zone": 1})
            eval_trials.append({"Trial": 1, "Strategy": i, "Start zone": 0})
    elif data_source == "random":
        # load random data
        trials, csv_read = pseudo_task_config(num_trials)
        eval_trials = []
        for i in range(2, 4):
            eval_trials.append({"Trial": 1, "Strategy": i, "Start zone": 1})
            eval_trials.append({"Trial": 1, "Strategy": i, "Start zone": 2})
    elif data_source == "block":
        # load random data
        trials, csv_read, eval_trials = block_task(num_trials)
    else:
        # let it run default n trials
        trials, csv_read = alternating_task(num_trials)
        eval_trials = []
        for i in range(2, 4):
            eval_trials.append({"Trial": 1, "Strategy": i, "Start zone": 1})
            eval_trials.append({"Trial": 1, "Strategy": i, "Start zone": 2})
        # eval_trials = [
        #     {"Trial": 1, "Strategy": 2, "Start zone": 1},
        #     {"Trial": 2, "Strategy": 3, "Start zone": 2},
        # ]

    return trials, csv_read, eval_trials


def pseudo_task_config(num_trials):
    df = pd.DataFrame(columns=["Trial", "Strategy", "Start zone"])
    trial_numbers = list(np.arange(1, num_trials + 1))
    df["Trial"] = trial_numbers
    strategy_list = [2, 3]  # , 3, 4]
    startzone_list = [1, 2]
    # k = number of items to select
    df["Strategy"] = random.choices(strategy_list, k=num_trials)
    df["Start zone"] = random.choices(startzone_list, k=num_trials)
    trials = df.T.to_dict().values()
    csv_read = df

    return list(trials), csv_read


def block_task(num_trials):
    df = pd.DataFrame(columns=["Trial", "Strategy", "Start zone"])
    trial_numbers = list(np.arange(1, num_trials + 1))
    df["Trial"] = trial_numbers
    strategy_list = ([2] * 200 + [3] * 200) * 40
    startzone = ([1] * 25 + [2] * 25 + [1] * 25 + [2] * 25) * 20 * 8
    eval_trials = []
    for i in range(2, 4):
        eval_trials.append({"Trial": 1, "Strategy": i, "Start zone": 1})
        eval_trials.append({"Trial": 1, "Strategy": i, "Start zone": 2})
    # k = number of items to select
    df["Strategy"] = strategy_list
    df["Start zone"] = startzone
    trials = df.T.to_dict().values()
    csv_read = df

    return list(trials), csv_read, eval_trials


def alternating_task(num_trials):
    df = pd.DataFrame(columns=["Trial", "Strategy", "Start zone"])
    trial_numbers = list(np.arange(1, num_trials + 1))
    df["Trial"] = trial_numbers
    strategy_list = [2] * num_trials  # , 2, 3, 4]
    # strategy_list = [2, 3] * num_trials  # , 2, 3, 4]
    startzone_0 = [0] * (num_trials // 2)
    startzone_1 = [1] * (num_trials - len(startzone_0))
    startzone = [None] * num_trials
    startzone[::2] = startzone_0
    startzone[1::2] = startzone_1
    # k = number of items to select
    df["Strategy"] = strategy_list
    df["Start zone"] = startzone
    trials = df.T.to_dict().values()
    csv_read = df

    return list(trials), csv_read
