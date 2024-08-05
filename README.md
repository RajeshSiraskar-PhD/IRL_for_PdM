# Inverse Reinforcement Learning for Predictive Maintenance

- 03-Aug-2024
- Learn the reward function from expert demonstrations

- V.1.0 04-Aug-2024: Create a new custom Gym

## PLAN / TO-DO
1. Environment modifications:
   - In ```reset```: Select a random tool-wear point - **but** in the first 10% of the tool-life "beginning" 
        ```self.current_time_step = np.random.randint(0, int(0.2 * self.records), 1, dtype=int)```
        
