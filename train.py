import warnings
from pathlib import Path

import hydra
import numpy as np
import torch
import wandb

import utils
from logger import Logger
from replay_buffer import make_replay
from video import VideoRecorder
from wrappers import make_env

warnings.filterwarnings("ignore", category=DeprecationWarning)

torch.backends.cudnn.benchmark = True


class Workspace:
    def __init__(self, cfg):
        self.work_dir = Path.cwd()
        print(f"workspace: {self.work_dir}")

        self.cfg = cfg

        # utils.set_seed_everywhere(cfg.seed)
        device = torch.device(cfg.device)

        # create logger
        if cfg.use_wandb:
            exp_name = "_".join(
                [
                    cfg.experiment,
                    cfg.animals[0],
                    cfg.data_source,
                    cfg.agent.name,
                    cfg.task,
                    str(cfg.seed),
                ]
            )
            wandb.init(project="continual_rl", group=cfg.agent.name, name=exp_name)
        self.logger = Logger(self.work_dir, use_tb=cfg.use_tb, use_wandb=cfg.use_wandb)

        # create envs
        self.train_env = make_env(cfg.task)
        self.eval_env = make_env(cfg.task)

        # create replay buffer
        # depending on agent select right replay buffer
        self.replay_buffer = make_replay(
            agent=cfg.agent.name,
            capacity=cfg.replay_buffer_size,
            sequence_length=cfg.sequence_length,
            device=cfg.device,
        )

        replay_iter = None

        self.video = VideoRecorder(
            self.work_dir if cfg.save_video else None,
            render_size=64,
            use_wandb=cfg.use_wandb,
        )

        self.agent = hydra.utils.instantiate(
            cfg.agent,
            num_obs=9,  # 82,  # 25,
            num_actions=3,
        )
        # initialize from pretrained
        if cfg.snapshot_ts > 0:
            pretrained_agent_model = self.load_snapshot()
            self.agent.init_from(pretrained_agent_model)
            print("agent loaded")

        self.activations = utils.Activations(
            self.work_dir if cfg.save_activations else None,
            self.agent if cfg.save_activations else None,
        )

        self.trials, self.csv_read, self.eval_trials = utils.load_trial_data(
            num_trials=cfg.trials, data_source=cfg.data_source, csv_path=cfg.csv_path
        )

        self.agent_stats = utils.PerformanceStats(
            self.work_dir if cfg.save_heatmaps_and_stats else None,
            csv_read=self.csv_read,
        )

        self.timer = utils.Timer()
        self._global_step = 0
        self._global_episode = 0
        # self.random_list = np.random.randint(0, 200, 180)
        # self.random_list = np.random.randint(0, 50, 50)
        self.random_list = np.random.choice(200, 50, replace=False)
        self.freq_counter = {"allo": 0, "ego": 0, "other": 0}

    @property
    def global_step(self):
        return self._global_step

    @property
    def trial_number(self):
        return self._trial_number

    def eval(self):
        # random_list = np.random.randint(0, 200, 100)
        self.random_list = np.random.choice(200, 50, replace=False)
        # freq_counter = {"allo": 0, "ego": 0, "other": 0}
        rew_counter = [0, 0, 0, 0]
        idx = 0
        orig_weights = self.agent.model.fc1.weight.data.clone()
        for trial in range(len(self.eval_trials)):
            observation = self.eval_env.reset(self.eval_trials[trial])
            done = False
            eval_return = 0
            step_count = 0
            self.video.init(self.eval_env, enabled=True)
            # fig, ax1 = plt.subplots(figsize=(10, 10))
            # sns.color_palette("colorblind", as_cmap=True)
            # sns.heatmap(self.agent.model.fc1.weight.data.cpu(), cmap="hot", ax=ax1)
            # plt.show()
            # input(...)
            while not done:
                with torch.no_grad():  # , utils.eval_mode(agent):

                    action, _, _ = self.agent.act(
                        observation,
                        self.global_step,
                        eval_mode=True,
                        eps=0.1,
                        trial_info=self.eval_trials[trial],
                    )
                observation, reward, done, info = self.eval_env.step(action)
                # import ipdb; ipdb.set_trace()
                # self.eval_env.render()
                self.video.record(self.eval_env)
                eval_return += reward
                step_count += 1

            rew_counter[idx] = eval_return
            idx += 1

            self.video.save(f"{self.global_step}_{trial}.mp4")
            print(eval_return)

        with self.logger.log_and_dump_ctx(self.global_step, ty="eval") as log:
            # log("episode_return", eval_return)
            log("episode_return_allo1", rew_counter[0])
            log("episode_return_allo2", rew_counter[1])
            log("episode_return_ego1", rew_counter[2])
            log("episode_return_ego2", rew_counter[3])
            # log("episode", episode)
            log("step", step_count)
            log("total_time", self.timer.total_time())

        if (rew_counter[0] == 1.0) and (rew_counter[1] == 1.0):
            self.freq_counter["allo"] += 1
        if (rew_counter[2] == 1.0) and (rew_counter[3] == 1.0):
            self.freq_counter["ego"] += 1
        else:
            self.freq_counter["other"] += 1

    def train(self):
        for animal in self.cfg.animals:

            episode, episode_step, episode_return = 0, 0, 0
            block_rewards, block_episode = 0, 0
            # total_step = 1
            total_rewards = 0
            observation = self.train_env.reset(self.trials[0])
            done = False
            reward = 0.0
            self.agent.reset(self.replay_buffer, step=self.trials[0])
            # replay_buffer.add()
            metrics = None
            prev_strategy = self.trials[0]["Strategy"]
            for trial in range(len(self.trials)):
                # n = trial % 2
                n = self.trials[trial]["Start zone"] - 1
                self._trial_number = episode
                observation = self.train_env.reset(self.trials[trial])
                self.activations.reset()
                # self.activations.add_activation(self.trials[trial])
                # self.activations.reset()
                episode_step, episode_return = 0, 0
                info = {"agent_pos": "[4 1]"}
                self.agent.reset(
                    self.replay_buffer, step=self.trials[trial], reward=reward
                )
                if (
                    self.trials[trial]["Strategy"] in [1, 2]
                    and prev_strategy not in [1, 2]
                ) or (
                    self.trials[trial]["Strategy"] in [3, 4]
                    and prev_strategy not in [3, 4]
                ):
                    print("switch")
                    self.agent.reset(
                        self.replay_buffer,
                        step=self.trials[trial],
                        reward=reward,
                        update=True,
                    )
                    self.replay_buffer.reset()
                    block_rewards = 0
                    block_episode = 0
                    prev_strategy = self.trials[trial]["Strategy"]
                # import ipdb
                # ipdb.set_trace()
                for step in range(self.cfg.num_train_steps + 1):
                    if done:
                        episode += 1
                        block_episode += 1
                        total_rewards += episode_return
                        block_rewards += episode_return
                        if metrics is not None:
                            elapsed_time, total_time = self.timer.reset()
                            with self.logger.log_and_dump_ctx(
                                self.global_step, ty="train"
                            ) as log:
                                log("fps", episode_step / elapsed_time)
                                log("total_time", total_time)
                                log("episode_return", episode_return)
                                log("avg_rewards", total_rewards / episode)
                                log("avg_block_rewards", block_rewards / block_episode)
                                log("strategy", n)
                                log("t_step", episode_step)
                                log("episode", episode)
                                log("buffer_size", len(self.replay_buffer))

                        done = False
                        # self.replay_buffer.reset()
                        # try to save snapshot
                        # if episode in self.cfg.snapshots:
                        #     self.save_snapshot()

                        self.agent_stats.record_outcome(
                            info, self.trials[trial], self.train_env.get_current_env()
                        )
                        break

                    with torch.no_grad():  # , utils.eval_mode(agent):
                        action, val, action_info = self.agent.act(
                            observation,
                            step,
                            eval_mode=False,
                            eps=0.0,  # 0.1,
                            trial_info=self.trials[trial],
                        )

                    # comment for testing -> no updates
                    if trial >= self.cfg.num_seed_steps:
                        # print("update")
                        metrics = self.agent.update(
                            self.replay_buffer, self.global_step, self.trials[trial]
                        )
                        self.logger.log_metrics(metrics, self.global_step, ty="train")

                    prev_observation = observation
                    # self.activations.add_activation(
                    #     info["agent_pos"], self.trials[trial]
                    # )
                    self.agent_stats.record_values(
                        info["agent_pos"],
                        val.cpu().detach().numpy()[0],
                        self.trials[trial],
                        episode_step,
                    )

                    observation, reward, done, info = self.train_env.step(action)
                    self.agent_stats.record_position(
                        info["agent_pos"], self.trials[trial]
                    )
                    # self.agent_stats.record_actions(
                    #     action_info, self.trials[trial], episode_step
                    # )
                    # if trial > 100:
                    # self.train_env.render()
                    episode_return += reward
                    self.replay_buffer.push(
                        prev_observation, action, reward, observation
                    )  # comment this for testing
                    episode_step += 1
                    # total_step += 1
                    self._global_step += 1

                # if episode % self.cfg.eval_every_steps == 0:
                #     # print("eval")
                #     self.eval()

                # # if trial > 2:
                # #     self.activations.add_activation(None, self.trials[trial])
                # # self.activations.reset()
                if (trial % 400 == 1) and (trial > 2):
                    # self.save_snapshot(trial)
                    # self.activations.save()
                    self.agent_stats.save_and_plot_heatmaps(trial)
                    # self.agent_stats.save_ego_allo()
                    # self.agent_stats.save_actions()
                    self.agent_stats.save_values()
                #     self.agent_stats.save_and_plot_heatmaps(trial)

            # self.save_snapshot(trial)
            self.activations.save()
            self.agent_stats.save_and_plot_heatmaps(trial)
            self.agent_stats.save_ego_allo()

    def save_snapshot(self, trial):
        snapshot_dir = self.work_dir / Path(self.cfg.snapshot_dir)
        snapshot_dir.mkdir(exist_ok=True, parents=True)
        print("Saving snapshot to:", snapshot_dir)
        snapshot = snapshot_dir / f"snapshot_{trial}.pt"
        payload = {
            "model_state_dict": self.agent.model.state_dict(),
        }
        with snapshot.open("wb") as f:
            torch.save(payload, f)

    def load_snapshot(self):
        snapshot_base_dir = Path(self.cfg.snapshot_base_dir)
        snapshot_dir = snapshot_base_dir

        def try_load(seed):
            snapshot = snapshot_dir / f"snapshot_{self.cfg.snapshot_ts}.pt"
            print(snapshot)
            if not snapshot.exists():
                return None
            with snapshot.open("rb") as f:
                payload = torch.load(f)
            return payload

        # try to load current seed
        payload = try_load(self.cfg.seed)
        if payload is not None:
            return payload
        # otherwise try random seed
        while True:
            seed = np.random.randint(1, 11)
            payload = try_load(seed)
            if payload is not None:
                return payload
        return None


@hydra.main(config_path=".", config_name="config")
def main(cfg):
    from train import Workspace as W

    root_dir = Path.cwd()
    workspace = W(cfg)
    snapshot = root_dir / "snapshot.pt"
    if snapshot.exists():
        print(f"resuming: {snapshot}")
        workspace.load_snapshot()
    workspace.train()


if __name__ == "__main__":
    main()
