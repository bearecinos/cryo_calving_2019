# OGGM cryo calving paper 2019

This repository contains the scripts used to implement a
frontal ablation parameterisation into [OGGM](www.oggm.org) and produce the 
results of
the paper submitted to The Cryosphere Disscus: 
[Recinos, et al,. in review](https://doi.org/10.5194/tc-2018-254)

This repository uses a OGGM version V1.1 pinned here: 
https://github.com/OGGM/oggm/releases/tag/v1.1

The content of the repository is the following: 

**I. cryo_cluster_scripts** (scripts used in a cluster environment)

1. Columbia Glacier runs
2. Calving vs Volume experiment
3. Sensitivity experiments with OGGM default parameters:
    **k**, **Glen A** and **fs** 
4. Alaska volume and volume below sea level calculated with 
different model configurations.

5. Marine-terminating glaciers frontal ablation fluxes calculated 
by correcting the depth and width of the terminus. (read
 [Recinos, et al,. in review](https://doi.org/10.5194/tc-2018-254) for more
 details).

6. Velocity experiment 

* To run the scripts make sure you have this repository on 
the home directory of your cluster environment.
* To run the scripts you **must follow the order of the folders.** 
* All scripts require a modified version of **RGI v.6.0**, where four 
*Marine-terminating glaciers* have been merged with their respective branches. 
**This file is located 
[here](https://cluster.klima.uni-bremen.de/~bea/cryo_calving_input_data/).**  

* RGI id's of the glacier merged:    

RGI60-01.21008 with RGI60-01.26732         
RGI60-01.10689 with RGI60-01.23635         
RGI60-01.26736 with RGI60-01.14443    
RGI60-01.03890 with RGI60-01.23664     

* The merge was necessary in order to use 
McNabb *et al.* 2014 Terminus positions data base.

* Scripts in folders **2, 3, 4, 5 and 6** require the output of 
the Columbia Glacier pre-processing run. 
**You can't run these folders without having ran the scripts 
in folder # 1.** 

**II. cryo_plots** (scripts for producing the figures)
* You will be able to run these scripts after you have completed the runs
in cryo_cluster_scripts and copied to your PC home directory the output_data
 folder generated from the runs. 
 The right path to copy this folder is in:
   `~/cryo_calving_2018/output_data/`

**Input data** 
* The input data to run all the scripts can be found 
[here](https://cluster.klima.uni-bremen.de/~bea/cryo_calving_input_data/)

**Output data**
* This folder will be generated once you have ran the 
first script of the folder cryo_cluster_scripts.
* Contains the output files for each folder in the cryo_cluster_scripts.

#### Please read the top of the scripts to know more about the ouput of each run.

## External libraries that are not in the OGGM env:    

The `rgi_overview.py` requiere additional libraries outside 
OGGM libraries such as:    
`collections` , `cartopy`, 
`cartopy.mpl.gridliner` , `cartopy.feature`, 
`cartopy.io.shapereader` and `cartopy.crs`   
 
 


