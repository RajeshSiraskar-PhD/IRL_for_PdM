import numpy as np
import pygame

import gymnasium as gym
from gymnasium import spaces


class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, size=5):
        self.size = size  # The size of the square grid
        self.window_size = 512  # The size of the PyGame window

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(0, size - 1, shape=(2,), dtype=int),
                "target": spaces.Box(0, size - 1, shape=(2,), dtype=int),
            }
        )

        # We have 4 actions, corresponding to "right", "up", "left", "down"
        self.action_space = spaces.Discrete(4)

        """
        The following dictionary maps abstract actions from `self.action_space` to
        the direction we will walk in if that action is taken.
        I.e. 0 corresponds to "right", 1 to "up" etc.
        """
        self._action_to_direction = {
            0: np.array([1, 0]),
            1: np.array([0, 1]),
            2: np.array([-1, 0]),
            3: np.array([0, -1]),
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

# %%
# Constructing Observations From Environment States
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Since we will need to compute observations both in ``reset`` and
# ``step``, it is often convenient to have a (private) method ``_get_obs``
# that translates the environment’s state into an observation. However,
# this is not mandatory and you may as well compute observations in
# ``reset`` and ``step`` separately:

    def _get_obs(self):
        return {"agent": self._agent_location, "target": self._target_location}

# %%
# We can also implement a similar method for the auxiliary information
# that is returned by ``step`` and ``reset``. In our case, we would like
# to provide the manhattan distance between the agent and the target:

    def _get_info(self):
        return {
            "distance": np.linalg.norm(
                self._agent_location - self._target_location, ord=1
            )
        }

# %%
# Oftentimes, info will also contain some data that is only available
# inside the ``step`` method (e.g. individual reward terms). In that case,
# we would have to update the dictionary that is returned by ``_get_info``
# in ``step``.

    # Reset
    # 1. Called to initiate a new episode and when 'Done'
    # 2. Assume that the ``step`` method will not be called before ``reset``
    # 3. Recommended to use RNG ``self.np_random`` provided by base class
    # 4. ** Important ** Must call ``super().reset(seed=seed)`` to correctly seed the RNG -- once done, we can randomly set the
    # state of our environment. In our case, we randomly choose the agent’s spatial location of "tool wear" 
    # 5. Must return a tuple of the *initial* observation - use ``_get_obs`` and ``_get_info`` 

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Choose the agent's location uniformly at random
        self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)

        # We will sample the target's location randomly until it does not coincide with the agent's location
        self._target_location = self._agent_location
        while np.array_equal(self._target_location, self._agent_location):
            self._target_location = self.np_random.integers(
                0, self.size, size=2, dtype=int
            )

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    # %%
    # Step
    # ~~~~
    #
    # The ``step`` method usually contains most of the logic of your
    # environment. It accepts an ``action``, computes the state of the
    # environment after applying that action and returns the 5-tuple
    # ``(observation, reward, terminated, truncated, info)``. See
    # :meth:`gymnasium.Env.step`. Once the new state of the environment has
    # been computed, we can check whether it is a terminal state and we set
    # ``done`` accordingly. Since we are using sparse binary rewards in
    # ``GridWorldEnv``, computing ``reward`` is trivial once we know
    # ``done``.To gather ``observation`` and ``info``, we can again make
    # use of ``_get_obs`` and ``_get_info``:

    def step(self, action):
        # Map the action (element of {0,1,2,3}) to the direction we walk in
        direction = self._action_to_direction[action]
        # We use `np.clip` to make sure we don't leave the grid
        self._agent_location = np.clip(
            self._agent_location + direction, 0, self.size - 1
        )
        # An episode is done iff the agent has reached the target
        terminated = np.array_equal(self._agent_location, self._target_location)
        reward = 1 if terminated else 0  # Binary sparse rewards
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info

