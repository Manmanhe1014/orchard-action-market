import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import random
from enum import Enum, auto
from orchard.algorithms import despawn_apple, despawn_apple_selfless_orchard, \
    spawn_apple, \
    spawn_apple_selfless_orchard, spawn_dirt

"""
The Orchard environment. Includes provisions for transition actions, spawning, and despawning.

action_algo: an algorithm that is used to process actions (agent movements). Defaults to just updating the environment from the singular agent action.
"""

#
# class EuclideanRewardsMixin:
#     """
#     Expects:
#       - self.n : int
#       - self.agents_list : iterable with .position
#       - calc_distance : function in scope
#     """
#     def _route_rewards(self, picker_id: int, c) -> np.ndarray:
#         res = np.zeros(self.n)
#         if c.consumed:
#             for agent_num in range(len(self.agents_list)):
#                 res[agent_num] = calc_distance(
#                     self.agents_list[agent_num].position, c.apple_pos
#                 )
#             if np.sum(res) == 0:
#                 return res
#             res = res / np.sum(res)
#             res = 2 * res
#             res[picker_id] = -1
#         return res


def calc_distance(pos1, pos2):
    return np.linalg.norm(pos1 - pos2)


class ActionMixin:
    """Mixin providing common methods for action enums."""
    @property
    def vector(self):
        """Return the action vector as a NumPy array."""
        return np.array(self._vector, dtype=np.int8)

    @property
    def idx(self):
        """Return the numeric index of the action."""
        return self._idx

    @property
    def one_hot(self):
        """Return a one-hot encoding of the action."""
        n = len(self.__class__)
        oh = np.zeros(n, dtype=np.float32)
        oh[self.idx] = 1.0
        return oh

    @classmethod
    def from_idx(cls, idx):
        """Lookup action by its numeric index."""
        try:
            return next(a for a in cls if a.idx == idx)
        except StopIteration:
            raise ValueError(f"Invalid action index: {idx}")


class Action1D(ActionMixin, Enum):
    LEFT_APPLE = (0, [0, -1])
    LEFT_DIRT = (1, [0, -1])
    RIGHT_APPLE = (0, [0, 1])
    RIGHT_DIRT = (1, [0, 1])
    STAY = (2, [0, 0])

    def __init__(self, action_type, vector):
        self._action_type = action_type  # 0=apple, 1=dirt, 2=stay
        self._vector = vector
        self._idx = len(self.__class__.__members__)  # auto-increment

    @property
    def action_type(self):
        return self._action_type


class Action2D(ActionMixin, Enum):
    LEFT_APPLE = (0, [0, -1])
    LEFT_DIRT = (1, [0, -1])
    RIGHT_APPLE = (0, [0, 1])
    RIGHT_DIRT = (1, [0, 1])
    STAY = (2, [0, 0])
    UP_APPLE = (0, [-1, 0])
    UP_DIRT = (1, [-1, 0])
    DOWN_APPLE = (0, [1, 0])
    DOWN_DIRT = (1, [1, 0])

    def __init__(self, action_type, vector):
        self._action_type = action_type
        self._vector = vector
        self._idx = len(self.__class__.__members__)

    @property
    def action_type(self):
        return self._action_type

@dataclass
class ProcessAction:
    reward_vector: np.ndarray
    picked: bool


@dataclass
class ConsumeResult:
    consumed: bool
    owner_id: Optional[int] = None

@dataclass
class ConsumeAppleResult(ConsumeResult):
    apple_pos: np.ndarray = None

@dataclass
class ConsumeDirtResult(ConsumeResult):
    dirt_pos: np.ndarray = None


class Orchard(ABC):
    def __init__(self,
                 length,
                 width,
                 num_agents,
                 agents_list=None,
                 action_algo=None,
                 spawn_algo=spawn_apple,
                 despawn_algo=despawn_apple,
                 spawn_dirt_algo=spawn_dirt,
                 s_target=0.1,
                 apple_mean_lifetime=None,
                 debug=False):
        self.length = length
        self.width = width
        self.available_actions = Action1D if width == 1 else Action2D

        self.n = num_agents
        assert agents_list is not None and self.n == len(agents_list)
        self.agents_list = agents_list

        # state grids
        self.agents = np.zeros((self.width, self.length), dtype=int)
        self.apples = np.zeros((self.width, self.length), dtype=int)
        self.dirt = np.zeros((self.width, self.length), dtype=int)

        # plug-ins
        self.spawn_algorithm = spawn_algo
        self.despawn_algorithm = despawn_algo
        self.spawn_dirt_algorithm = spawn_dirt_algo
        self.action_algorithm = action_algo or self.process_action

        # stats
        self.total_apples = 0
        self.apples_despawned = 0
        self.total_picked = 0
        self.total_dirt = 0

        # spawn/despawn rates (unchanged logic)
        self.spawn_rate = (self.n / (self.length * self.width)) * s_target
        if apple_mean_lifetime is None:
            self.despawn_rate = min(
                1.0,
                1.0 / ((math.ceil(1 / (2 * math.sqrt((self.n / (self.length * self.width))))) + 1)
                       if self.width > 1 else (1 / (2 * (self.n / (self.length * self.width))) + 1))
            )
        else:
            self.despawn_rate = 1 / apple_mean_lifetime

        # rendering flags
        self._rendering_initialized = False
        self.render_mode = None

        self.debug = debug
        if self.debug:
            self.state_history = []

        self.dummy_counter = 0

    def initialize(self, agents_list, agent_pos=None, apples=None):
        self.agents = np.zeros((self.width, self.length), dtype=int)
        self.apples = np.zeros((self.width, self.length), dtype=int) if apples is None else apples
        self.agents_list = agents_list
        self.set_positions(agent_pos)

    def set_positions(self, agent_pos=None):
        for i in range(self.n):
            position = agent_pos[i] if agent_pos is not None else np.random.randint(0, [self.width, self.length])
            self.agents_list[i].position = position
            self.agents[position[0], position[1]] += 1

    def get_state(self):
        return {"agents": self.agents.copy(), "apples": self.apples.copy(), "dirt": self.dirt.copy()}

    def _init_render(self):
        from rendering import Viewer
        self.viewer = Viewer((self.width, self.length))
        self._rendering_initialized = True

    def render(self):
        if not self._rendering_initialized:
            self._init_render()
        return self.viewer.render(self, return_rgb_array=self.render_mode == "rgb_array")

    @abstractmethod
    def _consume_apple(self, pos: np.ndarray) -> ConsumeAppleResult:
        """
        Mutate self.apples to reflect consumption.
        Return owner_id if applicable, else None.
        """

    @abstractmethod
    def _consume_dirt(self, pos: np.ndarray) -> ConsumeDirtResult:
        """Return owner_id if applicable, else None.
        Mutate self.dirt to reflect consumption.
        """

    @abstractmethod
    def _route_rewards(self, picker_id: int, owner_id: Optional[int]) -> np.ndarray:
        """Return (picker_reward, owner_id_to_credit_or_None)."""


    def _apply_move(self, position, action_vector):
        vec = np.asarray(action_vector, dtype=np.int8)
        new_pos = np.clip(position + vec, [0, 0], [self.width - 1, self.length - 1])
        self.agents[new_pos[0], new_pos[1]] += 1
        self.agents[position[0], position[1]] -= 1
        return new_pos



    def process_action(self, agent_id: int, 
                    position: np.ndarray, 
                    action) -> ProcessAction:
        if self.debug:
            self.state_history.append(
                np.concatenate([self.get_state()["agents"], self.get_state()["apples"], self.get_state()["dirt"]])
        )

    # Convert action index to enum if needed
        if isinstance(action, int):
            action = self.available_actions.from_idx(action)
    
    # Get the action type and vector from the enum
        action_type = action._action_type  # This is the consume type (0=apple, 1=dirt, 2=stay)
        action_vector = action.vector  # This is [dx, dy]
    
    # Apply movement
        if not np.array_equal(action_vector, [0, 0]):
            new_pos = self._apply_move(position, action_vector)
        else:
            new_pos = position
    
        self.agents_list[agent_id].position = new_pos

    # Consume based on action type
        c = ConsumeResult(consumed=False)
        if action_type == 0:  # apple action
            c = self._consume_apple(new_pos)
        elif action_type == 1:  # dirt action
            c = self._consume_dirt(new_pos)

        reward_vector = self._route_rewards(agent_id, c)
        return ProcessAction(reward_vector=reward_vector, picked=c.consumed)
    

    @abstractmethod
    def calculate_ir(self, position, action_vector, communal=True, agent_id=None):
        raise NotImplementedError


    def spawn_despawn(self):
        self.despawn_algorithm(self, self.despawn_rate)
        total_dirt = int(np.count_nonzero(self.dirt))
    
        self.spawn_algorithm(self, self.spawn_rate, total_dirt)
    
        self.spawn_dirt_algorithm(self, self.spawn_rate)

    @abstractmethod
    def get_sum_apples(self):
        raise NotImplementedError

    def process_action_eval(self, agent_id: int, position: np.ndarray, action):
        # Convert action index to enum if needed
        if isinstance(action, int):
            action = self.available_actions.from_idx(action)
    
        # Get the action type and vector from the enum
        action_type = action._action_type
        action_vector = action.vector
    
        # Apply movement
        if not np.array_equal(action_vector, [0, 0]):
            new_pos = self._apply_move(position, action_vector)
        else:
            new_pos = position

        self.agents_list[agent_id].position = new_pos

        # Consume based on action type (same as training)
        if action_type == 0:  # apple action
            self._consume_apple(new_pos)
        elif action_type == 1:  # dirt action
            self._consume_dirt(new_pos)

    def remove_apple(self, pos: np.ndarray):
        pass


class OrchardBasic(Orchard):
    def _consume_apple(self, pos: np.ndarray) -> ConsumeAppleResult:
        if self.apples[pos[0], pos[1]] > 0:
            self.apples[pos[0], pos[1]] -= 1
            return ConsumeAppleResult(consumed=True)
        return ConsumeAppleResult(consumed=False)

    def _consume_dirt(self, pos: np.ndarray) -> ConsumeDirtResult:
        if self.dirt[pos[0], pos[1]] > 0:
            self.dirt[pos[0], pos[1]] -= 1
            self.total_dirt += 1  # Did you add this?
            return ConsumeDirtResult(consumed=True)
        return ConsumeDirtResult(consumed=False)
        
    def _route_rewards(self, picker_id: int, c: ConsumeResult) -> np.ndarray:
        res = np.zeros(self.n)
        if not c.consumed:
            return res
        if isinstance(c, ConsumeAppleResult):
            res[picker_id] = 1.0
        elif isinstance(c, ConsumeDirtResult):
            res[picker_id] = -1.0
        return res

    def calculate_ir(self, position, action_vector, communal=True, agent_id=None):
        new_position = np.clip(position + action_vector, [0, 0], self.agents.shape - np.array([1, 1]))
        agents = self.agents.copy()
        apples = self.apples.copy()
        dirt = self.dirt.copy()  # ← ADD THIS
        agents[new_position[0], new_position[1]] += 1
        agents[position[0], position[1]] -= 1
        if apples[new_position[0], new_position[1]] > 0:
            apples[new_position[0], new_position[1]] -= 1
            return 1, agents, apples, dirt, new_position  # ← ADD dirt
        else:
            return 0, agents, apples, dirt, new_position  # ← ADD dirt

    def get_sum_apples(self):
        return np.sum(self.apples)


class OrchardWithAppleIDs(Orchard):
    def __init__(self,
                 length,
                 width,
                 num_agents,
                 agents_list=None,
                 action_algo=None,
                 spawn_algo=spawn_apple_selfless_orchard,
                 despawn_algo=despawn_apple_selfless_orchard,
                 s_target=0.1,
                 apple_mean_lifetime=None):
        super().__init__(length, width, num_agents, agents_list, 
                         action_algo, spawn_algo, despawn_algo, 
                         s_target, apple_mean_lifetime)

    def _consume_apple(self, pos: np.ndarray) -> ConsumeAppleResult:
        owner_plus1 = self.apples[pos[0], pos[1]]
        if owner_plus1 > 0:
            self.apples[pos[0], pos[1]] = 0
            return ConsumeAppleResult(consumed=True, owner_id=owner_plus1.item())
        return ConsumeAppleResult(consumed=False)
    
    def _consume_dirt(self, pos: np.ndarray) -> ConsumeDirtResult:
        owner_plus1 = self.dirt[pos[0], pos[1]]
        if owner_plus1 > 0:
            self.dirt[pos[0], pos[1]] = 0
            return ConsumeDirtResult(consumed=True, owner_id=owner_plus1.item())
        return ConsumeDirtResult(consumed=False)

    def calculate_ir(self, position, action_vector, communal=True, agent_id=None):
        new_position = np.clip(position + action_vector, [0, 0], self.agents.shape - np.array([1, 1]))
        agents = self.agents.copy()
        apples = self.apples.copy()
        dirt = self.dirt.copy()  # ← ADD THIS
        agents[new_position[0], new_position[1]] += 1
        agents[position[0], position[1]] -= 1
        if apples[new_position[0], new_position[1]] > 0:
            apple_id = apples[new_position[0], new_position[1]]
            apples[new_position[0], new_position[1]] = 0
            if communal or ((not communal) and (agent_id + 1) == apple_id):
                return 1, agents, apples, dirt, new_position  # ← ADD dirt
            else:
                return 0, agents, apples, dirt, new_position  # ← ADD dirt
        else:
            return 0, agents, apples, dirt, new_position  # ← ADD dirt

    @abstractmethod
    def _route_rewards(self, picker_id: int, c: ConsumeAppleResult):
        raise NotImplementedError

    def get_sum_apples(self):
        return np.count_nonzero(self.apples)


class OrchardIDs(OrchardWithAppleIDs):
    def _route_rewards(self, picker_id: int, c: ConsumeResult):
        if not c.consumed:
            return 0, None, 0
        if isinstance(c, ConsumeDirtResult):
            return -1, None, 0   # picker −1, no owner
        # existing apple logic:
        if c.owner_id == (picker_id + 1):
            return 0, c.owner_id, 0
        else:
            return 0, c.owner_id, 1


class OrchardMineNoReward(OrchardWithAppleIDs):
    def _route_rewards(self, picker_id: int, c: ConsumeResult):
        if not c.consumed:
            return 0, None, 0
        if isinstance(c, ConsumeDirtResult):
            return -1, None, 0           # ← add this
        if c.owner_id == (picker_id + 1):
            return 0, c.owner_id, 0
        else:
            return 0, c.owner_id, 1


class OrchardSelfless(OrchardWithAppleIDs):
    def _route_rewards(self, picker_id: int, c: ConsumeResult):
        if not c.consumed:
            return 0, None, 0
        if isinstance(c, ConsumeDirtResult):
            return -1, None, 0           # ← add this
        if c.owner_id == (picker_id + 1):
            return 1, c.owner_id, 1
        else:
            return 0, c.owner_id, 1


class OrchardMineAllRewards(OrchardWithAppleIDs):
    def _route_rewards(self, picker_id: int, c: ConsumeResult):
        if not c.consumed:
            return 0, None, 0
        if isinstance(c, ConsumeDirtResult):
            return -1, None, 0
        return 1, c.owner_id, 1


class OrchardEuclideanRewards(OrchardBasic):
    def _consume_apple(self, pos: np.ndarray) -> ConsumeAppleResult:
        if self.apples[pos[0], pos[1]] > 0:
            self.apples[pos[0], pos[1]] -= 1
            return ConsumeAppleResult(consumed=True, apple_pos=pos)
        return ConsumeAppleResult(consumed=False)

    def _consume_dirt(self, pos: np.ndarray) -> ConsumeDirtResult:
        if self.dirt[pos[0], pos[1]] > 0:
            self.dirt[pos[0], pos[1]] -= 1
            self.total_dirt += 1 
            return ConsumeDirtResult(consumed=True, dirt_pos=pos)
        return ConsumeDirtResult(consumed=False)
    
    def _route_rewards(self, picker_id: int, c: ConsumeResult) -> np.ndarray:
        res = np.zeros(self.n, dtype=float)
        if not c.consumed:
            return res

        if isinstance(c, ConsumeAppleResult):
            # existing distance-based split
            dists = np.array([
                calc_distance(self.agents_list[i].position, c.apple_pos)
                for i in range(self.n)
            ], dtype=float)
            s = dists.sum()
            if s > 0:
                res = dists / s
            # else res stays zeros
            return res

        elif isinstance(c, ConsumeDirtResult):
            # simple penalty to the acting agent
            res[picker_id] = -1.0
            return res


class OrchardEuclideanNegativeRewards(OrchardEuclideanRewards):
    def _route_rewards(self, picker_id: int, c: ConsumeResult) -> np.ndarray:
        res = np.zeros(self.n, dtype=float)
        if not c.consumed:
            return res

        if isinstance(c, ConsumeAppleResult):
            dists = np.array([
                calc_distance(self.agents_list[i].position, c.apple_pos)
                for i in range(self.n)
            ], dtype=float)
            s = dists.sum()
            if s > 0:
                res = 2.0 * (dists / s)
            res[picker_id] = -1.0
            return res

        elif isinstance(c, ConsumeDirtResult):
            res[picker_id] = -1.0
            return res


class OrchardBasicNewDynamic(OrchardBasic):
    def _consume_apple(self, pos: np.ndarray) -> ConsumeAppleResult:
        if self.apples[pos[0], pos[1]] > 0:
            return ConsumeAppleResult(consumed=True)
        return ConsumeAppleResult(consumed=False)

    def remove_apple(self, pos: np.ndarray):
        if self.apples[pos[0], pos[1]] > 0:
            self.apples[pos[0], pos[1]] -= 1
            self.total_picked += 1

    def calculate_ir(self, position, action_vector, communal=True, agent_id=None):
        new_position = np.clip(position + action_vector, [0, 0], self.agents.shape - np.array([1, 1]))
        agents = self.agents.copy()
        apples = self.apples.copy()
        dirt = self.dirt.copy()  # ← ADD THIS
        agents[new_position[0], new_position[1]] += 1
        agents[position[0], position[1]] -= 1
        if apples[new_position[0], new_position[1]] > 0:
            apple_id = apples[new_position[0], new_position[1]]
            if communal or ((not communal) and (agent_id + 1) == apple_id):
                return 1, agents, apples, dirt, new_position  # ← ADD dirt
            else:
                return 0, agents, apples, dirt, new_position  # ← ADD dirt
        else:
            return 0, agents, apples, dirt, new_position  # ← ADD dirt


class OrchardEuclideanRewardsNewDynamic(OrchardEuclideanRewards):
    def _consume_apple(self, pos: np.ndarray) -> ConsumeAppleResult:
        if self.apples[pos[0], pos[1]] > 0:
            return ConsumeAppleResult(consumed=True, apple_pos=pos)
        return ConsumeAppleResult(consumed=False)
    
    def _consume_dirt(self, pos: np.ndarray) -> ConsumeDirtResult:
        if self.dirt[pos[0], pos[1]] > 0:
            return ConsumeDirtResult(consumed=True, dirt_pos=pos)
        return ConsumeDirtResult(consumed=False)

    def calculate_ir(self, position, action_vector, communal=True, agent_id=None):
        new_position = np.clip(position + action_vector, [0, 0], self.agents.shape - np.array([1, 1]))
        agents = self.agents.copy()
        apples = self.apples.copy()
        dirt = self.dirt.copy()
        agents[new_position[0], new_position[1]] += 1
        agents[position[0], position[1]] -= 1
        if apples[new_position[0], new_position[1]] > 0:
            apple_id = apples[new_position[0], new_position[1]]
            if communal or ((not communal) and (agent_id + 1) == apple_id):
                return 1, agents, apples, dirt, new_position
            else:
                return 0, agents, apples, dirt, new_position
        else:
            return 0, agents, apples, dirt, new_position

    def remove_apple(self, pos: np.ndarray):
        if self.apples[pos[0], pos[1]] > 0:
            self.apples[pos[0], pos[1]] -= 1
            self.total_picked += 1


class OrchardEuclideanNegativeRewardsNewDynamic(OrchardEuclideanNegativeRewards):
    def _consume_apple(self, pos: np.ndarray) -> ConsumeAppleResult:
        if self.apples[pos[0], pos[1]] > 0:
            return ConsumeAppleResult(consumed=True, apple_pos=pos)
        return ConsumeAppleResult(consumed=False)
    
    def _consume_dirt(self, pos: np.ndarray) -> ConsumeDirtResult:
        if self.dirt[pos[0], pos[1]] > 0:
            return ConsumeDirtResult(consumed=True, dirt_pos=pos)
        return ConsumeDirtResult(consumed=False)

    def calculate_ir(self, position, action_vector, communal=True, agent_id=None):
        new_position = np.clip(position + action_vector, [0, 0], self.agents.shape - np.array([1, 1]))
        agents = self.agents.copy()
        apples = self.apples.copy()
        dirt = self.dirt.copy()  # ← ADD THIS
        agents[new_position[0], new_position[1]] += 1
        agents[position[0], position[1]] -= 1
        if apples[new_position[0], new_position[1]] > 0:
            apple_id = apples[new_position[0], new_position[1]]
            if communal or ((not communal) and (agent_id + 1) == apple_id):
                return 1, agents, apples, dirt, new_position  # ← ADD dirt
            else:
                return 0, agents, apples, dirt, new_position  # ← ADD dirt
        else:
            return 0, agents, apples, dirt, new_position  # ← ADD dirt

    def remove_apple(self, pos: np.ndarray):
        if self.apples[pos[0], pos[1]] > 0:
            self.apples[pos[0], pos[1]] -= 1
            self.total_picked += 1
