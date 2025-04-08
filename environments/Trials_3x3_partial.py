import random

import numpy as np
import torch
from gym_minigrid.minigrid import *
from gym_minigrid.register import register


class TrialEnv(MiniGridEnv):
    """
    Empty grid environment, no obstacles, sparse reward

    0 - EMPTY
    1 - WALL
    2 - REWARD
    3 - AGENT

    """

    DIR_UP = 3
    DIR_DOWN = 1
    DIR_LEFT = 2
    DIR_RIGHT = 4

    NORTH = (4, 1)
    SOUTH = (4, 1)  # (4, 7)
    EAST = (7, 4)
    WEST = (1, 4)

    def __init__(
        self,
        agent_pos,
        goal_position,
        cue=False,
        size=9,
        partial=True,
        t_maze=False,
        use_reward_as_cue=False,
        hyper_params=None,
        task_id=None,
        random_cues=False,
    ):
        self.agent_start_pos = agent_pos
        self.agent_start_dir = (
            self.DIR_DOWN
        )  # if agent_pos == self.SOUTH else self.DIR_DOWN
        # goal_position = (1, 5)
        self.goal_position = goal_position
        self.partial = partial
        self.t_maze = t_maze
        self.use_reward_as_cue = use_reward_as_cue
        self.use_key = hyper_params["use_key_cue"]
        self.use_box = hyper_params["use_box_cue"]
        self.use_ball = hyper_params["use_ball_cue"]
        self.use_lava = hyper_params["use_lava_cue"]
        self.random_cues = random_cues
        self.flag_cue_removal = False
        if self.random_cues:
            self.use_key = np.random.choice([True, False], p=[1.00, 0.00])
            self.use_box = np.random.choice([True, False], p=[1.00, 0.00])
            self.use_ball = np.random.choice([True, False], p=[1.00, 0.00])
            self.use_lava = np.random.choice([True, False], p=[1.00, 0.00])
        self.once_been = False
        self.task_id = task_id
        if self.partial:
            self.agent_view_size = 3
        else:
            self.agent_view_size = 7

        self.cue = cue

        if goal_position[0] == 1:
            self.opp_goal_position = (7, goal_position[1])
        elif goal_position[0] == 7:
            self.opp_goal_position = (1, goal_position[1])
        super().__init__(
            grid_size=size, max_steps=200, agent_view_size=self.agent_view_size
        )

    def _gen_grid(self, width, height):

        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate borders
        self.grid.wall_rect(0, 0, width, height)

        # Generate PLUS
        for x in [1, 2, 3, 5, 6, 7]:
            self.grid.vert_wall(x, 1, 3)

        # self.grid.vert_wall(0,0,9)
        # self.grid.horz_wall(0,8,9)
        # self.grid.vert_wall(1,2,2)
        # for x in [1]:
        #     self.grid.vert_wall(x, 5, 3)
        # self.grid.horz_wall(2,7,2)
        # self.grid.horz_wall(5,7,3)

        for x in [1, 2, 3, 5, 6, 7]:
            self.grid.vert_wall(x, 5, 3)

        if self.t_maze:
            # Generate T-shaped
            # if self.agent_start_pos == self.SOUTH:
            #     self.grid.vert_wall(4, 1, 3)
            # else:
            self.grid.vert_wall(4, 5, 3)
            # self.grid.vert_wall(4, 6, 3)

        # for x in [1, 2, 6, 7]:
        #     self.put_obj(Wall(), x, 4)

        # Place goal
        self.put_obj(Goal(), self.goal_position[0], self.goal_position[1])
        self.put_obj(Op_Goal(), self.opp_goal_position[0], self.opp_goal_position[1])

        if self.cue:
            if self.task_id in [1, 2, 5, 6]:
                # Place box
                if self.use_key:
                    self.put_obj(Key(color="red"), 3, 1)
                    # self.put_obj(Key(color="red"), 3, 3)
                if self.use_box:
                    self.put_obj(Box(color="yellow"), 5, 1)
                    # self.put_obj(Box(color="yellow"), 5, 3)
                # if self.use_ball:
                #     self.put_obj(Ball(color="purple"), 3, 7)
                # if self.use_lava:
                #     self.put_obj(Lava(), 5, 7)
            elif self.task_id in [0, 3, 4, 7]:
                # if self.use_key:
                #     self.put_obj(Key(color="red"), 5, 7)
                # if self.use_box:
                #     self.put_obj(Box(color="yellow"), 3, 7)
                if self.use_ball:
                    self.put_obj(Ball(color="purple"), 5, 1)
                    # self.put_obj(Ball(color="purple"), 5, 3)
                if self.use_lava:
                    self.put_obj(Lava(), 3, 1)
                    # self.put_obj(Lava(), 3, 3)
            else:
                print("Error: not a valid task id")

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        self.mission = "get to the green goal square"

        grid_mask = np.array([s != None for s in self.grid.grid])
        self.grid_data = np.zeros([self.grid.width * self.grid.height])
        self.grid_data[grid_mask] = -1
        if self.use_reward_as_cue:
            self.grid_data[
                self.goal_position[1] * self.grid.width + self.goal_position[0]
            ] = 2
        else:
            self.grid_data[
                self.goal_position[1] * self.grid.width + self.goal_position[0]
            ] = 0
            self.grid_data[
                self.opp_goal_position[1] * self.grid.width + self.opp_goal_position[0]
            ] = 0
            # print(self.grid_data)
        if self.cue:
            # distractor cues
            # self.grid_data[21] = 80
            # self.grid_data[23] = 80
            if self.task_id in [1, 2, 5, 6]:
                if self.use_key:
                    self.grid_data[12] = 5  # + np.random.normal(0, 15)  # key value
                    # self.grid_data[30] = 5  # key value
                if self.use_box:
                    self.grid_data[14] = 60  # + np.random.normal(0, 15)  # box value
                    # self.grid_data[32] = 60  # box value
                # if self.use_ball:
                #     self.grid_data[66] = 30  # ball value
                # if self.use_lava:
                #     self.grid_data[68] = 90  # lava value
            elif self.task_id in [0, 3, 4, 7]:
                if self.use_lava:
                    self.grid_data[12] = 90  # + np.random.normal(0, 15)  # lava value
                    # self.grid_data[30] = 90
                if self.use_ball:
                    self.grid_data[14] = 30  # + np.random.normal(0, 15)  # ball value
                    # self.grid_data[32] = 30
                # if self.use_box:
                #     self.grid_data[66] = 60
                # if self.use_key:
                #     self.grid_data[68] = 5

    def getStateSize(self):
        if self.partial:
            return self.agent_view_size * self.agent_view_size  # + 1

        return self.grid.width * self.grid.height + 1

    def extractState(self):

        # make cues random over same trial

        # if self.random_cues:
        #     self.use_key = np.random.choice([True, False], p=[0.10, 0.90])
        #     self.use_box = np.random.choice([True, False], p=[0.10, 0.90])
        #     self.use_ball = np.random.choice([True, False], p=[0.10, 0.90])
        #     self.use_lava = np.random.choice([True, False], p=[0.10, 0.90])
        # if self.cue:
        #     if self.task_id in [1, 2, 5, 6]:
        #         self.grid_data[12] = 5 if self.use_key else -1
        #         self.grid_data[14] = 60 if self.use_box else -1
        #     elif self.task_id in [0, 3, 4, 7]:
        #         self.grid_data[12] = 90 if self.use_lava else -1
        #         self.grid_data[14] = 30 if self.use_ball else -1

        state = np.copy(self.grid_data)
        # print(state)
        # state[0] = 99

        # Update agent position and direction
        # if agent_dir is 0 replace with 4 to differentiate from corridor
        if self.agent_dir == 0:
            agent_dir = 4
        else:
            agent_dir = self.agent_dir
        state[
            self.agent_pos[1] * self.grid.width + self.agent_pos[0]
        ] = agent_dir  # 3 don't give agent direction as it has to be inferred
        # print('agent_dir', self.agent_dir)
        # print('state --', state)
        if (state[40] > 0) and self.flag_cue_removal:
            # remove cues
            self.flag_cue_removal = False
            # print("removing cues")
            if self.random_cues:
                self.use_key = np.random.choice([True, False], p=[0.00, 1.00])
                self.use_box = np.random.choice([True, False], p=[0.00, 1.00])
                self.use_ball = np.random.choice([True, False], p=[0.00, 1.00])
                self.use_lava = np.random.choice([True, False], p=[0.00, 1.00])
            if self.cue:
                if self.task_id in [1, 2, 5, 6]:
                    self.grid_data[12] = 5 if self.use_key else -1
                    self.grid_data[14] = 60 if self.use_box else -1
                elif self.task_id in [0, 3, 4, 7]:
                    self.grid_data[12] = 90 if self.use_lava else -1
                    self.grid_data[14] = 30 if self.use_ball else -1

        # return torch.FloatTensor([state])

        ### partial view from agent point of view
        part_state = np.zeros(9)
        current_agent_pos = self.agent_pos[1] * self.grid.width + self.agent_pos[0]
        partial_view_size = self.agent_view_size
        l0 = current_agent_pos - 20
        l1 = current_agent_pos - 10
        l4 = current_agent_pos + 10
        l5 = current_agent_pos + 20
        # print(state)
        if agent_dir == 1:  # down
            part_state[0:3] = state[
                current_agent_pos - 1 : current_agent_pos - 1 + partial_view_size
            ]
            part_state[3:6] = state[l4 + 1 - partial_view_size : l4 + 1]
            part_state[6:9] = state[l5 - partial_view_size : l5]
        elif agent_dir == 4:  # right
            part_state[0:3] = state[l1 + 1 : l1 + partial_view_size + 1]
            part_state[3:6] = state[
                current_agent_pos : current_agent_pos + partial_view_size
            ]
            part_state[6:9] = state[l4 + 1 - partial_view_size + 1 : l4 + 1 + 1]
        elif agent_dir == 2:  # left
            part_state[0:3] = state[l1 - 1 : l1 + partial_view_size - 1]
            part_state[3:6] = state[
                current_agent_pos - 2 : current_agent_pos - 2 + partial_view_size
            ]
            part_state[6:9] = state[l4 - partial_view_size : l4]
        elif agent_dir == 3:  # up
            part_state[0:3] = state[l0 + 1 : l0 + partial_view_size + 1]
            part_state[3:6] = state[l1 : l1 + partial_view_size]
            part_state[6:9] = state[
                current_agent_pos - 1 : current_agent_pos + partial_view_size - 1
            ]

        # return torch.FloatTensor([state])
        return torch.FloatTensor([part_state])
        # return torch.FloatTensor([p_state])

    def get_offset(self):
        if self.agent_start_pos == self.NORTH:
            if self.goal_position == self.WEST:
                return 10
            else:
                return -10

        else:
            if self.goal_position == self.WEST:
                return 50
            else:
                return -50

    def _reward(self):
        """
        Compute the reward to be given upon success
        """
        reward = 1.0  # - 0.9 * ((self.step_count)/ self.max_steps)
        return reward
