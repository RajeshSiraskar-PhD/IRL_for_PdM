# ------------------------------------------------------------------------------------------------------------------------
# Milling tool environment
# V.1.0 04-Aug-2024
# V.1.1 05-Aug-2024: Added random tool wear start point (random within 10% records from the begining)
# V.2.0 07-Aug-2024: Drop time - one of the main indicators. Add all force (x, y, z) and vibration (x, y, z) 
# ------------------------------------------------------------------------------------------------------------------------

import numpy as np
import gymnasium as gym
from gymnasium import spaces

LAMBDA = 0.01
RANDOM_TOOL_START_OF_LIFE = 0.10 # 10% from the start
NO_ACTION = 0
REPLACE = 1

# Information arrays 
a_time = []
a_actions = []
a_action_text = []
a_rewards = []
a_rul = []
a_cost = []
a_replacements = []
a_time_since_last_replacement = []
a_action_recommended = []

class MillingTool_Env(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, records=0, rul_threshold=0.0):
        print(f'\n -- Milling tool environment initiatlized. Potential records {records}. RUL threshold {rul_threshold:4.3f}')
        # Initialize
        self.df = None
        self.current_time_step = 0
        self.records = records
        self.maintenance_cost = 0.0
        self.replacement_events = 0
        self.time_since_last_replacement = 0
        
        self.rul_threshold = rul_threshold # Usually 5% from 0.0 i.e. 95th percentile record value from the very end 
        
        # Observation vector: ['timestamp', 'vibration_x', 'vibration_y', 'force_z', 'tool_wear', 'RUL', 'ACTION_CODE']
        # PHM Dataset: force_x	force_y	force_z	vibration_x	vibration_y	vibration_z	acoustic_emission_rms	tool_wear	ACTION_CODE	RUL

        high = np.array(
            [
                1.0,          # Max. force_x
                1.0,          # Max. force_y
                1.0,          # Max. force_z
                1.0,          # Max. vibration_x
                1.0,          # Max. vibration_y
                1.0,          # Max. vibration_z                
            ],
            dtype=np.float32,
        )

        # observation space lower limits
        low = np.array(
            [
                -1.0,          # Max. force_x
                -1.0,          # Max. force_y
                -1.0,          # Max. force_z
                -1.0,          # Max. vibration_x
                -1.0,          # Max. vibration_y
                -1.0,          # Max. vibration_z
            ],
            dtype=np.float32,
        )

        self.observation_space = spaces.Box(low, high, dtype=np.float32)

        # Actions - Normal, L1-maintenance, L2-maintenance, Replace
        self.action_space = spaces.Discrete(2)

    ## Add tool wear data-set
    def tool_wear_data(self, df):
        self.df = df
        self.records = len(df.index)
        print(f'\n - Milling tool environment: Tool wear data updated: {self.records}')
        
    ## Constructing Observations From Environment States
    # - Observations are needed for both ``reset`` and ``step``, 
    # - Create private method ``_get_obs`` that translates the environment’s state into an observation.
    # - One can additionally use _get_info (in step and reset) if some auxilliary info. needs to be sent - for e.g. Expert action or Reward      #   info. or even RUL
    def _get_observation(self):
        if (self.df is not None):
            obs_values = np.array([
                self.df.loc[self.current_time_step, 'force_x'],
                self.df.loc[self.current_time_step, 'force_y'],
                self.df.loc[self.current_time_step, 'force_z'],
                self.df.loc[self.current_time_step, 'vibration_x'],
                self.df.loc[self.current_time_step, 'vibration_y'],
                self.df.loc[self.current_time_step, 'vibration_y']
            ], dtype=np.float32)
        else:
            obs_values = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        
        
        observation = obs_values.flatten()
        return observation

    # Get the current RUL reading, note this is NOT part of the observation
    def _get_auxilliary_info(self):
        if (self.df is not None):
            # From database extract recommended action
            recommended_action = int(self.df.loc[self.current_time_step, 'ACTION_CODE'])
            rul =  float(self.df.loc[self.current_time_step, 'RUL'])
        else:
            # No database - use dummy values
            recommended_action = 0
            rul = 0.0

        return recommended_action, rul
            
    ## Reset
    # 1. Called to initiate a new episode and when 'Done'
    # 2. Assume that the ``step`` method will not be called before ``reset``
    # 3. Recommended to use RNG ``self.np_random`` provided by base class
    # 4. ** Important ** Must call ``super().reset(seed=seed)`` to correctly seed the RNG -- once done, we can randomly set the
    # state of our environment. In our case, we randomly choose the agent’s spatial location of "tool wear" 
    # 5. Must return a tuple of the *initial* observation - use ``_get_observation`` 

    def reset(self, seed=None, options=None):
        # print(' - reset')
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Choose the tool wear at a random time (spatial) location from a uniformly random distribution
        self.current_time_step = np.random.randint(0, int(RANDOM_TOOL_START_OF_LIFE * self.records), 1, dtype=int)
        observation = self._get_observation()
        info = {'reset':'Reset'}
        
        return observation, info

    ## Step
    # 1. Method the logic environment.
    # 2. Accepts an ``action``, computes the state of the environment **after** applying that action
    # 3. returns the 5-tuple ``(observation, reward, terminated, truncated, info)``
    # 4. Once the new state of the environment has been computed - check terminal state / set rewards
    # 5. To gather ``observation`` and ``info``, we can use of ``_get_obs`` and ``_get_info``:

    def step(self, action):
        # print(f' - step {action}')
        terminated = False
        reward = 0.0
        info = {'Step':'-'}
        # Get auxilliary info: current RUL reading (note this is NOT part of the observation) and the expert's recommended action
        recommended_action, self.rul = self._get_auxilliary_info()
        self.maintenance_cost = 0.0
        self.replacement_events += 0
        
        if self.current_time_step >= self.records:
            terminated = done = True
            info = {'Step':'EOF'}
        elif self.rul <= self.rul_threshold: # Less-than-equal 0 (or near zero)
            terminated = done = True            
            info = {'Step':'RUL threshold crossed'}
        elif action == NO_ACTION: # Normal state
            self.current_time_step += 1
            # 1% reduction in life
            self.maintenance_cost += 0.1
            info = {'Step':'None'}
        elif action == REPLACE:
            self.current_time_step += 1
            # Replace the tool - reset to begining - but to a random position in the first 10% time-steps 
            self.maintenance_cost += 10.0
            self.replacement_events += 1
            self.time_since_last_replacement = self.current_time_step
            info = {'Step':'* REPLACE *'}

        # Action taken, set reward    
        self.reward = (self.current_time_step + 1) / (self.maintenance_cost+LAMBDA)
        self.reward = self.reward / 1e3

        # Information arrays 
        a_time.append(self.current_time_step)
        a_actions.append(action)
        a_action_text.append(recommended_action)
        a_rewards.append(self.reward)
        a_rul.append(self.rul)
        a_cost.append(self.maintenance_cost)
        a_replacements.append(self.replacement_events)
        a_time_since_last_replacement.append(self.time_since_last_replacement)
        a_action_recommended.append(recommended_action)
        
        # Action taken, reward set for that action, now take in next observation
        reward = float(self.reward)
        observation = self._get_observation()
        
        if self.render_mode == "human":
            print('{0:<20} | RUL: {1:>8.2f} | Cost: {2:>8.2f} | Reward: {3:>12.3f}'.format(action_text, self.rul, self.maintenance_cost, self.reward))

        return observation, reward, terminated, False, info