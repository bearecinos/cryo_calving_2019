# Configuration experiments 

These scripts will calculate the total glacier volume of Alaska with different model
configurations.

| Configuration  | Description                                                                 |
| -------------: | :--------------------------------------------------------------------------:|
| 1              |  k<sub>1</sub> = 0.68515, Glen A = OGGM default, f<sub>s</sub> = 0.0         |
| 2              |  k<sub>2</sub> = 0.7899, Glen A = OGGM default, f<sub>s</sub> = 0.0          |
| 3              |  k<sub>1</sub> = 0.68515, Glen A = OGGM default, f<sub>s</sub> = OGGM default|
| 4              |  k<sub>2</sub> = 0.7899, Glen A = OGGM default, f<sub>s</sub> = OGGM default |
| 5              |  k<sub>1</sub> = 0.68515, Glen A = 2.92e10−24, f<sub>s</sub> = 0.0           |
| 6              |  k<sub>2</sub> = 0.7899, Glen A = 4.04e10−24, f<sub>s</sub> = 0.0            |
| 7              |  k<sub>1</sub> = 0.68515, Glen A = 2.58e10−24 , f<sub>s</sub> = OGGM default |
| 8              |  k<sub>2</sub> = 0.7899, Glen A = 3.69e10−24, f<sub>s</sub> = OGGM default   |

The content of each experiment (each folder) is the following:

4.1 No calving experiments:

4.2 With calving experiments only MT

4.3 With calving experiments only MT (vbsl):   

These scripts are exactly the same as 4.2, with the following differences: 

* **vbsl** stands for volume below sea level and in here we replace the 
inversion output name with: `inversion_onput_without_calving.pkl`   

* If we do this inside the scripts of folder 4.2 we will get the wrong 
`glacier_characteristics.csv`, and then the wrong volume 
after accounting for frontal ablation. This due to the fact that 
the OGGM function: `utils.compile_glacier_statistics()` does not record 
the volume after calving or after the `filesuffix='_with_calving_'` name has been 
added to the `inversion_output.pkl` 

To execute runs 4_1, 4_2, 4_3 in the cluster type these commands in experiment root 
folder: *4_Runs_different_configurations*:  

`./run_no_calving_exp.sh`   
`./run_with_calving_exp.sh`   
`./run_with_calving_exp_vbsl.sh`   
