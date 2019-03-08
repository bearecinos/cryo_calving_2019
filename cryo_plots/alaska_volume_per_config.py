# This script will calculate the total alaska volume per experiment

import oggm.cfg as cfg
import numpy as np
import pandas as pd
import os
import glob
import seaborn as sns
os.getcwd()
import matplotlib.pyplot as plt
from matplotlib import rcParams

# Useful functions to calculate volume and
# volume conversions

def calculate_sea_level_equivalent(value):
    """
    Calculates sea level equivalent of a volume
    of ice in km^3
    taken from: http://www.antarcticglaciers.org

    :param value: glacier volume
    :return: glacier volume in s.l.e
    """
    # Convert density of ice to Gt/km^3
    rho_ice = 900 * 1e-3 # Gt/km^3

    area_ocean = 3.618e8 # km^2
    height_ocean = 1e-6 # km (1 mm)

    # Volume of water required to raise global sea levels by 1 mm
    vol_water = area_ocean*height_ocean # km^3 of water

    mass_ice = value * rho_ice # Gt
    return mass_ice*(1 / vol_water)

def calculate_volume_percentage(volume_one, volume_two):
    return np.around((volume_two*100)/volume_one,2)-100

def read_experiment_file(filename):
    """
    Read glacier_characteristics.csv file
    for each model configuration

    :param filename:
    :return: total volume of that experiment
             the name of the file
             base directory
    """
    glacier_run = pd.read_csv(filename, index_col='rgi_id')
    tail = os.path.basename(filename)
    basedir = os.path.dirname(filename)
    total_volume = glacier_run['inv_volume_km3'].sum()
    return total_volume, tail, basedir

def read_experiment_file_vbsl(filename):
    """
    Reads volume_below_sea_level.csv
    for each model configuration
    :param filename:
    :return: total_volume: below sea level before calving
             total_volume_c: below sea level after calving
             tail: the name of the file
    """
    glacier_run = pd.read_csv(filename)
    tail = os.path.basename(filename)
    total_volume = glacier_run['volume bsl'].sum()
    total_volume_c = glacier_run['volume bsl with calving'].sum()
    return total_volume, total_volume_c, tail


# Reading the directories
MAIN_PATH = os.path.expanduser('~/cryo_calving_2019/')
plot_path = os.path.join(MAIN_PATH, 'plots/')
output_dir_path = os.path.join(MAIN_PATH,
                       'output_data/4_Runs_different_configurations/')


exclude = set(['4_3_With_calving_exp_onlyMT_vbsl'])

# Creating the paths for Marine glacier_char.csv files
full_exp_name = []

for path, subdirs, files in os.walk(output_dir_path, topdown=True):
    subdirs[:] = [d for d in subdirs if d not in exclude]
    subdirs[:] = [d for d in subdirs if "rest" not in d]
    subdirs[:] = sorted(subdirs)

    for name in files:
        full_exp_name.append(os.path.join(path,name))

# Extract volumes for non-calving experiments
# Reading no-calving experiments contained in 4_1_No_calving_exp folder
# print(full_exp_name[0:8])
volume_no_calving = []
exp_name = []
exp_number = [1, 2, 3, 4, 5, 6, 7, 8]

for f in full_exp_name[0:8]:
    volume, tails, basedir = read_experiment_file(f)
    volume_no_calving += [np.around(volume,2)]
    exp_name += [basedir]
print('Experiment name', exp_number)
print('Volume before calving', volume_no_calving)

volume_no_calving_sle = []
for value in volume_no_calving:
    volume_no_calving_sle.append(calculate_sea_level_equivalent(value))
print('Volume before calving SLE', volume_no_calving_sle)

# Extract volumes for calving experiments
# Reading calving experiments contained in 4_2_With_calving_exp_onlyMT
#print(full_exp_name[8:len(full_exp_name)])
volume_calving = []
exp_name_c = []
exp_number_c = [1, 2, 3, 4, 5, 6, 7, 8]

for f in full_exp_name[8:len(full_exp_name)]:
    volume, tails, basedir= read_experiment_file(f)
    volume_calving += [np.around(volume,2)]
    exp_name_c += [basedir]
print('Experiment name', exp_number_c)
print('Volume after calving', volume_calving)


volume_calving_sle = []
for value in volume_calving:
    volume_calving_sle.append(calculate_sea_level_equivalent(value))
print('Volume after calving SLE', volume_calving_sle)

# Extract volumes from volume_below_sea_level.csv
# for the different configurations
# Reading calving experiments contained in 4_3_With_calving_exp_onlyMT_vbsl
exp_dir_path = os.path.join(output_dir_path,
                            '4_3_With_calving_exp_onlyMT_vbsl')
dir_name = os.listdir(exp_dir_path)

full_dir_name = []
for d in dir_name:
    full_dir_name.append(os.path.join(exp_dir_path,
                             d +'/volume_below_sea_level.csv'))

full_dir_name = sorted(full_dir_name)
#print(full_dir_name)

# Reading no calving volumes below sea level
vbsl = []
vbsl_c = []
exp_name_bsl = []

for f in full_dir_name:
    volume, volume_c, tails = read_experiment_file_vbsl(f)
    vbsl += [np.around(volume,2)]
    vbsl_c += [np.around(volume_c,2)]
    exp_name_bsl += [tails]

print('Experiment', exp_number)
print('Volume bsl no calving', vbsl)
print('Volume bsl calving', vbsl_c)

# sea level equivalent
vbsl_sle = []
vbsl_c_sle = []

for i, j in zip(vbsl, vbsl_c):
    vbsl_sle.append(calculate_sea_level_equivalent(i))
    vbsl_c_sle.append(calculate_sea_level_equivalent(j))

print('Volume bsl no calving in s.l.e', vbsl_sle)
print('Volume bsl calving in s.l.e', vbsl_c_sle)

percentage = []
for i, j in zip(volume_no_calving, volume_calving):
    percentage.append(calculate_volume_percentage(i,j))

print('Percentage', percentage)

print('Percent for columbia', calculate_volume_percentage(270.40, 349.39))


d = {'Experiment No': exp_number,
     'Volume no calving in s.l.e': volume_no_calving_sle,
     'Volume no calving bsl in s.l.e': vbsl_sle,
     'Volume with calving in s.l.e': volume_calving_sle,
     'Volume with calving bsl in s.l.e': vbsl_c_sle,
     'Volume differences in s.l.e':
         [np.abs(a - b) for a, b in zip(volume_no_calving_sle,volume_calving_sle)],
     'Volume no calving in km3': volume_no_calving,
     'Volume no calving bsl km3': vbsl,
     'Volume with calving in km3': volume_calving,
     'Volume with calving bsl km3': vbsl_c,
     'Volume differences in km3':
         [np.abs(a - b) for a, b in zip(volume_no_calving,volume_calving)],
     }

ds = pd.DataFrame(data=d)

ds = ds.sort_values(by=['Experiment No'])
#ds.to_csv(os.path.join(plot_path,
#                       'MT_glaciers_volume_per_exp.csv'))
print('----------- For de paper ------------------')
print('Mean and std volume with calving',
      np.round(np.mean(volume_calving_sle),2), np.round(np.std(volume_calving_sle),2))
print('Mean and std volume no calving',
      np.round(np.mean(volume_no_calving_sle),2), np.round(np.std(volume_no_calving_sle),2))
print('Mean and std volume below sea level with calving',
      np.round(np.mean(vbsl_c_sle),2), np.round(np.std(vbsl_c_sle),2))

print('Mean and std volume below sea level without calving',
      np.round(np.mean(vbsl_sle),2), np.round(np.std(vbsl_sle),2))

print('TABLE',ds)

print('is vbsl bigger than differnces', vbsl_c > ds['Volume differences in km3'].values)

# Plot settings
# Set figure width and height in cm
width_cm = 12
height_cm = 8

fig = plt.figure(figsize=(width_cm, height_cm))
sns.set(style="white", context="talk")

ax1=fig.add_subplot(111)
ax2= ax1.twiny()

# Plot settings
rcParams['axes.labelsize'] = 20
rcParams['xtick.labelsize'] = 20
rcParams['ytick.labelsize'] = 20
rcParams['legend.fontsize'] = 14

N = len(ds)
ind = np.arange(N)
graph_width = 0.35
labels = ds['Experiment No'].values

bars1 = ds['Volume no calving bsl km3'].values
bars2 = ds['Volume no calving in km3'].values
#print(bars2)

bars3 = ds['Volume with calving bsl km3'].values
bars4 = ds['Volume with calving in km3'].values

sns.set_color_codes()
sns.set_color_codes("colorblind")

p1 = ax1.barh(ind, bars1*-1,
              color="indianred", edgecolor="white", height=graph_width)
p2 = ax1.barh(ind, bars2, height=graph_width, edgecolor="white")

p3 = ax1.barh(ind - graph_width, bars3*-1,
              color="indianred",edgecolor="white", height=graph_width)
p4 = ax1.barh(ind - graph_width, bars4, color=sns.xkcd_rgb["teal green"], edgecolor="white",
              height=graph_width)


ax1.set_xticks([-1000, 0, 1000, 2000, 3000, 4000, 5000])
ax1.set_xticklabels(abs(ax1.get_xticks()), fontsize=20)
ax2.set_xlim(ax1.get_xlim())
ax2.tick_params('Volume [mm SLE]', fontsize=20)
ax2.set_xticks(ax1.get_xticks())
array = ax1.get_xticks()
#print(array)

sle = []
for value in array:
    sle.append(np.round(abs(calculate_sea_level_equivalent(value)),2))
#print(sle)

ax2.set_xticklabels(sle,fontsize=20)
ax2.set_xlabel('Volume [mm SLE]', fontsize=18)

plt.yticks(ind - graph_width/2, labels)

ax1.set_yticklabels(labels, fontsize=20)
ax1.set_ylabel('Model configurations', fontsize=18)
ax1.set_xlabel('Volume [km³]',fontsize=18)
plt.legend((p2[0], p4[0], p1[0]),
           ('Volume without frontal ablation',
            'Volume with frontal ablation',
            'Volume below sea level'),
            frameon=True,
            bbox_to_anchor=(1.1, -0.15), ncol=3, fontsize=15)
plt.margins(0.05)

#plt.show()
plt.savefig(os.path.join(plot_path, 'marine_volume.pdf'),
            dpi=150, bbox_inches='tight')