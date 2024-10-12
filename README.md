# Reinforcement Learning for Predictive Maintenance

**Approach**: PhD Thesis work

V.1.0 11-Oct-2024:
- Create two environments: PHM and NUAA
- To demonstrate robustness - Use multiple data sets from each 3 + 3 = 6
- Additional: Add noise and Break-down
- Publish results


### Notes:

**Environment**: 
- Only **raw** features. NO derived features like Dr Sameer Sayyad.
- 

```
    high = np.array(  [
        1.0,          # Max. force_x
        1.0,          # Max. force_y
        1.0,          # Max. force_z
        1.0,          # Max. vibration_x
        1.0,          # Max. vibration_y
        1.0,          # Max. vibration_z                
    ], dtype=np.float32,) 
```