import numpy as np
from gym import spaces

# from neuronav.envs.grid_env import (GridEnv, GridObsType, GridSize,
#                                     OrientationType)
from neuronav.envs.grid_env import GridEnv

from environments.Trials_3x3_partial import TrialEnv

# from environments.Trials_3x3_NS import TrialEnv
# from environments.Trials_3x3_test import TrialEnv
# from environments.Trials_3x3_long_test import TrialEnv


# from neuronav.envs.grid_topographies import GridTopography


class ContinualWrapper(GridEnv):
    def __init__(self, cues_locs, reward_locs: None, **kwargs):
        super().__init__(**kwargs)
        self.cues_locs = cues_locs
        # self.goal_position = [1, 1]
        if reward_locs is not None:
            self.reward_locs = reward_locs
        # 5x5 field view
        self.observation_space = spaces.Box(
            0, self.state_size, shape=(5 * 5,), dtype=np.int32
        )

    @property
    def observation(self):
        obs = self.get_observation(self.agent_pos)
        obs = obs.sum(axis=2).flatten()
        return obs

    def reset(
        self,
        reward_locs=None,
        cue_locs=None,
        agent_pos=None,
        episode_length=100,
        random_start=False,
    ):
        self.done = False
        self.episode_time = 0
        self.orientation = 0
        self.max_episode_time = episode_length

        if agent_pos is not None:
            self.agent_pos = agent_pos
        elif random_start:
            self.agent_pos = self.get_free_spot()
        else:
            self.agent_pos = self.agent_start_pos

        if reward_locs is not None:
            self.reward_locs = reward_locs
        else:
            self.reward_locs = self.topo_reward_locs

        if cue_locs is not None:
            self.cues_locs = cue_locs

        return self.observation

    def getStateSize(self):
        return 25

    def grid(self, render_objects=True):
        grid = np.zeros([self.grid_size, self.grid_size, 3])
        if render_objects:
            grid[self.agent_pos[0], self.agent_pos[1], :] = self.orientation
            for loc, reward in self.reward_locs.items():
                if reward > 0:
                    grid[loc[0], loc[1], 1] = reward
                else:
                    grid[loc[0], loc[1], 0] = np.abs(reward)
        for block in self.blocks:
            grid[block[0], block[1], :] = 0.5
        grid = self.add_cues(grid, self.cues_locs)
        return grid

    def add_cues(self, grid, cue_locs=None):
        for loc, val in cue_locs.items():
            grid[loc[0], loc[1], :] = val

        return grid


class Neuro:
    def __init__(self, name, n_envs):
        # init
        self.name = name
        self.n_envs = n_envs
        self.env_id = 0
        self.val = [0.75, 0.75]
        self.rew_locs = [{(5, 2): 1.0, (5, 8): -1.0}, {(5, 2): -1.0, (5, 8): 1.0}]
        self.locs = {(4, 4), (4, 6), (6, 4), (6, 6)}
        self.cue_locs = lambda x: {loc: x for loc in self.locs}
        self.envs = []
        self.create_envs()

    def create_envs(self):
        for n in range(self.n_envs):
            env = ContinualWrapper(
                cues_locs=self.cue_locs(self.val[n]),
                reward_locs=self.rew_locs[n],
                topography=GridTopography.t_maze_cont,
                obs_type=GridObsType.window,
                orientation_type=OrientationType.dynamic,
            )
            self.envs.append(env)
            env = ContinualWrapper(
                cues_locs=self.cue_locs(self.val[n]),
                reward_locs=self.rew_locs[n],
                topography=GridTopography.t_maze_cont_N,
                obs_type=GridObsType.window,
                orientation_type=OrientationType.dynamic,
            )
            self.envs.append(env)

    def step(self, action):
        return self.envs[self.env_id].step(action)

    def reset(self, env_id):
        self.env_id = env_id
        observation = self.envs[env_id].reset(
            reward_locs=self.rew_locs[0],
            cue_locs=self.cue_locs(self.val[env_id]),
            episode_length=100,
        )
        return observation

    def render(self, mode="human"):
        return self.envs[self.env_id].render(mode)


class Minigrid:
    def __init__(self, name):
        self.name = name
        self.environments = []
        self.current_env = None
        self.hyper_params = {
            "use_key_cue": True,
            "use_box_cue": True,
            "use_ball_cue": True,
            "use_lava_cue": True,
        }
        self.scenarios = [
            # EGO-CENTRIC
            {
                "GOAL": TrialEnv.WEST,
                "AGENT_POS": TrialEnv.SOUTH,
                "CUE": True,
                "T_ID": 0,
            },  # 0, invert cues
            {
                "GOAL": TrialEnv.WEST,
                "AGENT_POS": TrialEnv.NORTH,
                "CUE": True,
                "T_ID": 1,
            },  # 1
            {
                "GOAL": TrialEnv.EAST,
                "AGENT_POS": TrialEnv.NORTH,
                "CUE": True,
                "T_ID": 2,
            },  # 2
            {
                "GOAL": TrialEnv.EAST,
                "AGENT_POS": TrialEnv.SOUTH,
                "CUE": True,
                "T_ID": 3,
            },  # 3, invert cues
            # ALLO-CENTRIC
            {
                "GOAL": TrialEnv.EAST,
                "AGENT_POS": TrialEnv.SOUTH,
                "CUE": True,
                "T_ID": 4,
            },  # 4, invert cues
            {
                "GOAL": TrialEnv.WEST,
                "AGENT_POS": TrialEnv.NORTH,
                "CUE": True,
                "T_ID": 5,
            },  # 5
            {
                "GOAL": TrialEnv.EAST,
                "AGENT_POS": TrialEnv.NORTH,
                "CUE": True,
                "T_ID": 6,
            },  # 6
            {
                "GOAL": TrialEnv.WEST,
                "AGENT_POS": TrialEnv.SOUTH,
                "CUE": True,
                "T_ID": 7,
            },  # 7, invert cues
        ]

        self.create_envs()

    def get_maze_config(self, strategy, start_zone):

        # START ZONE
        # 1 - NORTH
        # 2 - SOUTH

        # STRATEGY
        # 1/2 - ALLOCENTRIC
        # 3/4 - EGOCENTRIC

        if start_zone == 1:
            if strategy == 1:
                return 5
            if strategy == 2:
                return 6
            if strategy == 3:
                return 1
            if strategy == 4:
                return 2

        else:
            if strategy == 1:
                return 4
            if strategy == 2:
                return 7
            if strategy == 3:
                return 0
            if strategy == 4:
                return 3

        # wrong config
        return -1

    def create_envs(self):
        for scenario in self.scenarios:
            env = TrialEnv(
                agent_pos=scenario["AGENT_POS"],
                goal_position=scenario["GOAL"],
                cue=scenario["CUE"],
                partial=False,
                t_maze=True,
                use_reward_as_cue=False,
                hyper_params=self.hyper_params,
                task_id=scenario["T_ID"],
            )
            self.environments.append(env)

    def create_single_env(self, scenario):
        env = TrialEnv(
            agent_pos=scenario["AGENT_POS"],
            goal_position=scenario["GOAL"],
            cue=scenario["CUE"],
            partial=True,
            t_maze=True,
            use_reward_as_cue=False,
            hyper_params=self.hyper_params,
            task_id=scenario["T_ID"],
        )
        return env

    def get_current_env(self):
        return self.current_env

    def reset(self, trial=None):
        strategy = trial["Strategy"]
        start_zone = trial["Start zone"]
        config = self.get_maze_config(strategy, start_zone)
        self.current_env = self.create_single_env(self.scenarios[config])
        observation = self.current_env.reset()
        observation = self.current_env.extractState()
        observation = observation.squeeze(0).numpy()
        return observation

    def step(self, action):
        info = {}
        observation, reward, done, current_agent_pos, old_info = self.current_env.step(
            action
        )
        observation = self.current_env.extractState()
        observation = observation.squeeze(0).numpy()
        info["agent_pos"] = current_agent_pos
        info["old_info"] = old_info
        return observation, reward, done, info

    def render(self, mode="human"):
        return self.current_env.render(mode)


def make_env(name):
    if name == "neuro":
        env = Neuro("neuro", 2)
    elif name == "minigrid":
        env = Minigrid("minigrid")
    else:
        env = None

    return env
