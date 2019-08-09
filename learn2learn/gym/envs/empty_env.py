from enum import IntEnum

from gym_minigrid.minigrid import *
import cherry as ch


class EmptyEnv(MiniGridEnv):
    def __init__(
            self,
            size=7,
            agent_start_dir=0,
            task={}
    ):
        self._task = task
        start = task.get('start')
        if start == None:
            start = (int((size + 1) / 2), int((size + 1) / 2))

        goal = task.get('goal')
        if goal == None:
            goal = (size - 2, size - 2)

        self.agent_start_pos = start
        self.agent_start_dir = agent_start_dir
        self.agent_dir = agent_start_dir
        self.size = size
        self.goal_pos = goal

        super().__init__(
            grid_size=size,
            max_steps=4 * size * size,
            see_through_walls=True,
#            agent_view_size=3,
        )
        self.agent_dir = 0

        class Actions(IntEnum):
            left = 0
            right = 1
            up = 2
            down = 3

        self.actions = Actions
        self.action_space = spaces.Discrete(len(self.actions))

        width = size - 2
        height = size - 2

        self.action_size = len(self.actions)
        self.state_size = width * height

        self.observation_space = spaces.Discrete(width * height)
        self.reward_range = (-1 * (width + height - 2), 0)

    def _gen_grid(self, width, height):

        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)
        self.grid.set(self.goal_pos[0], self.goal_pos[1], Goal())

        self.start_pos = self.agent_start_pos
        self.start_dir = self.agent_start_dir
        self.start_dir = 0

        self.mission = "get to the green goal square"

    def sample_tasks(self, num_tasks):
        goals = (self.np_random.randint(2, size=[num_tasks, 2]) * (self.size - 3)) + 1
        goals[:, 0] = 1
        tasks = [{'goal': goal} for goal in goals]
        return tasks

    def reset_task(self, task):
        self._task = task
        self.goal_pos = task.get('goal')

    def reset(self):
        self._gen_grid(self.width, self.height)

        # These fields should be defined by _gen_grid
        assert self.start_pos is not None
        assert self.start_dir is not None

        # Check that the agent doesn't overlap with an object
        start_cell = self.grid.get(*self.start_pos)
        assert start_cell is None or start_cell.can_overlap()

        # Place agent
        self.agent_pos = self.start_pos
        self.aget_dir = self.start_dir

        self.carrying = None
        self.step_count = 0

        # Return first observation
        obs = ((self.width - 2) * (self.agent_pos[1] - 1) + self.agent_pos[0]) - 1
        return ch.onehot(obs, 81)

    def _reward(self):
        return -1 * self._manhattan_distance()

    def _manhattan_distance(self):
        return abs(self.agent_pos[0] - self.goal_pos[0]) + abs(self.agent_pos[1] - self.goal_pos[1])

    def step(self, action):
        self.step_count += 1

        reward = 0
        done = False

        agent_x_pos = self.agent_pos[0]
        agent_y_pos = self.agent_pos[1]

        # Get the possible new positions of the agent (0th and Nth columns and rows have walls)
        up_pos = (agent_x_pos, max(1, agent_y_pos - 1))
        down_pos = (agent_x_pos, min(self.height - 1, agent_y_pos + 1))
        left_pos = (max(1, agent_x_pos - 1), agent_y_pos)
        right_pos = (min(self.width - 1, agent_x_pos + 1), agent_y_pos)

        # Get the contents of the possible new cells 
        up_cell = self.grid.get(*up_pos)
        down_cell = self.grid.get(*down_pos)
        left_cell = self.grid.get(*left_pos)
        right_cell = self.grid.get(*right_pos)

        # ADD NOTE HERE

        if action == self.actions.left:
            if left_cell == None or left_cell.can_overlap():
                self.agent_pos = left_pos

        elif action == self.actions.right:
            if right_cell == None or right_cell.can_overlap():
                self.agent_pos = right_pos

        elif action == self.actions.up:
            if up_cell == None or up_cell.can_overlap():
                self.agent_pos = up_pos

        elif action == self.actions.down:
            if down_cell == None or down_cell.can_overlap():
                self.agent_pos = down_pos

        new_cell = self.grid.get(*self.agent_pos)

        if new_cell != None and new_cell.type == 'lava':
            done = True

        if new_cell != None and new_cell.type == 'goal':
            done = True

        if self.step_count >= self.max_steps:
            done = True

        obs = ((self.width - 2) * (self.agent_pos[1] - 1) + self.agent_pos[0]) - 1
        reward = self._reward()

        return obs, reward, done, self._task
