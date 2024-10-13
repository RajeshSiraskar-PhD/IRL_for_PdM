# Reinforcement Learning for Predictive Maintenance

## Important notes for Thesis and Presentation
- Only **raw** features. NO derived features like Dr Sameer Sayyad.
- Proof of generalization:
    - Fundamentally **different** datasets - material, sensors etc.
    - Features are **different**

## Important implementation notes
- Rewards for NUAA and PHM similar range: Keep number of records similar - bout 1000
- RUL value threshold should have some value and not 0


**Approach**: PhD Thesis work

V.1.0 11-Oct-2024:
- Create two environments: PHM and NUAA
- To demonstrate robustness - Use multiple data sets from each 3 + 3 = 6
- Additional: Add noise and Break-down
- Publish results


- [NUAA](https://ieee-dataport.org/open-access/tool-wear-dataset-nuaaideahouse)
- 

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

| PHM		  | Uniwear	    |
|-------------|-------------|
| time		  | time	    |
| force_x     | axial_force | 
| force_y     |             |			
| force_z     | force_z     | 
| vibration_x | vibration_x | 
| vibration_y | vibration_y | 
| vibration_z | vibration1  | 
| 			  | vibration2  |	
| tool_wear   | tool_wear   | 


| Dataset | Workpiece               | Tool                |
|---------|-------------------------|---------------------|
| PHM2010 | Stainless steel (HRC52)	| Tungsten Carbide    |
| NUAA    | Titanium (TC4)	        | Solid Carbide       |