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
exp_number = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]

for f in full_exp_name[0:13]:
    volume, tails, basedir = read_experiment_file(f)
    volume_no_calving += [np.around(volume,2)]
    exp_name += [basedir]

volume_no_calving_sle = []
for value in volume_no_calving:
    volume_no_calving_sle.append(calculate_sea_level_equivalent(value))

# Extract volumes for calving experiments
# Reading calving experiments contained in 4_2_With_calving_exp_onlyMT
volume_calving = []
exp_name_c = []
exp_number_c = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]

for f in full_exp_name[13:len(full_exp_name)]:
    volume, tails, basedir= read_experiment_file(f)
    volume_calving += [np.around(volume,2)]
    exp_name_c += [basedir]

volume_calving_sle = []
for value in volume_calving:
    volume_calving_sle.append(calculate_sea_level_equivalent(value))

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

# Reading no calving volumes below sea level
vbsl = []
vbsl_c = []
exp_name_bsl = []

for f in full_dir_name:
    volume, volume_c, tails = read_experiment_file_vbsl(f)
    vbsl += [np.around(volume,2)]
    vbsl_c += [np.around(volume_c,2)]
    exp_name_bsl += [tails]

# sea level equivalent
vbsl_sle = []
vbsl_c_sle = []

for i, j in zip(vbsl, vbsl_c):
    vbsl_sle.append(calculate_sea_level_equivalent(i))
    vbsl_c_sle.append(calculate_sea_level_equivalent(j))


percentage = []
for i, j in zip(volume_no_calving, volume_calving):
    percentage.append(calculate_volume_percentage(i,j))

print('FOR THE PAPER')
print('----------------')
print('Percentage', min(percentage), max(percentage))
print('Percent for columbia', calculate_volume_percentage(270.40, 349.39))
print('Gt equivalent columbia', 2.98161468959857/1.091)

# Make a dataframe with each configuration output
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
         [np.abs(a - b) for a, b in zip(volume_no_calving,volume_calving)]
     }
ds = pd.DataFrame(data=d)

ds = ds.sort_values(by=['Experiment No'])
ds.to_csv(os.path.join(plot_path,
                       'MT_glaciers_volume_per_exp.csv'))

print('----------- For de paper more information ------------------')

print('Mean and std volume with calving for all config',
      np.round(np.mean(volume_calving_sle),2),
      np.round(np.std(volume_calving_sle),2))

print('Mean and std volume no calving for all config',
      np.round(np.mean(volume_no_calving_sle),2),
      np.round(np.std(volume_no_calving_sle),2))

print('Mean and std volume below sea level with calving',
      np.round(np.mean(vbsl_c_sle),2),
      np.round(np.std(vbsl_c_sle),2))

print('Mean and std volume below sea level without calving',
      np.round(np.mean(vbsl_sle),2),
      np.round(np.std(vbsl_sle),2))

print('TABLE',ds)

print('For the paper check if the volume below sea level is bigger than diff among config.')
print(vbsl_c > ds['Volume differences in km3'].values)

diff_config = np.diff(volume_calving)
total = abs(diff_config / volume_calving[0:-1])*100

print('volume after calving differences between configs', total)


# Reading Farinotti 2019 regional volume for MT glaciers.
farinotti_data = pd.read_csv(os.path.join(MAIN_PATH,
                              'input_data/farinotti_volume.csv'))
#Sum the volumes in farinotti data
vol_fari = farinotti_data['vol_itmix_m3'].sum()*1e-9
vol_bsl_fari = farinotti_data['vol_bsl_itmix_m3'].sum()*1e-9


print('FOR THE PAPER')
print('our estimate after calving', np.round(np.mean(volume_calving_sle),2))
print('farinotti', calculate_sea_level_equivalent(vol_fari))
print('percentage of vol change',
      calculate_volume_percentage(np.mean(volume_calving),vol_fari))

# Plot everything!
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

N = len(ds)+1
ind = np.arange(N)
graph_width = 0.35
labels = np.append(ds['Experiment No'].values, 14)

bars1 = np.append(ds['Volume no calving bsl km3'].values, vol_bsl_fari)
bars2 = np.append(ds['Volume no calving in km3'].values, vol_fari)

bars3 = ds['Volume with calving bsl km3'].values
bars4 = ds['Volume with calving in km3'].values

sns.set_color_codes()
sns.set_color_codes("colorblind")

p1 = ax1.barh(ind[0:8], bars1[0:8]*-1,
              color="indianred", edgecolor="white", height=graph_width)

p1_extra = ax1.barh(ind[8:13], bars1[8:13]*-1,
              color="indianred", edgecolor="white", height=graph_width,
                    alpha=0.5)

p2 = ax1.barh(ind[0:8], bars2[0:8], color=sns.xkcd_rgb["ocean blue"],
              height=graph_width, edgecolor="white")

p2_extra = ax1.barh(ind[8:13], bars2[8:13], color=sns.xkcd_rgb["ocean blue"],
                    height=graph_width, edgecolor="white",
                    alpha=0.5)

p3 = ax1.barh(ind[0:8] - graph_width, bars3[0:8]*-1,
              color="indianred",edgecolor="white", height=graph_width)

p3_extra = ax1.barh(ind[8:13] - graph_width, bars3[8:13]*-1,
              color="indianred", edgecolor="white", alpha=0.5,
                    height=graph_width)

p4 = ax1.barh(ind[0:8] - graph_width, bars4[0:8], color=sns.xkcd_rgb["teal green"],
              edgecolor="white",
              height=graph_width)

p4_extra = ax1.barh(ind[8:13] - graph_width, bars4[8:13], color=sns.xkcd_rgb["teal green"],
              edgecolor="white", alpha=0.5,
              height=graph_width)

fari_low = ax1.barh(ind[-1]-graph_width,  bars1[13:14]*-1, color="indianred",
                edgecolor="white",
                height=graph_width)

fari = ax1.barh(ind[-1]-graph_width,  bars2[13:14], color=sns.xkcd_rgb["grey"],
                edgecolor="white", height=graph_width)

ax1.set_xticks([-1000, 0, 1000, 2000, 3000, 4000, 5000])
ax1.set_xticklabels(abs(ax1.get_xticks()), fontsize=20)

ax2.set_xlim(ax1.get_xlim())
ax2.set_xticks(ax1.get_xticks())
array = ax1.get_xticks()

# Get the other axis on sea level equivalent
sle = []
for value in array:
    sle.append(np.round(abs(calculate_sea_level_equivalent(value)),2))

ax2.set_xticklabels(sle,fontsize=20)
ax2.set_xlabel('Volume [mm SLE]', fontsize=18)

plt.yticks(ind - graph_width/2, labels)

ax1.set_yticklabels(labels[0:-1], fontsize=20)
ax1.set_ylabel('Model configurations', fontsize=18)
ax1.set_xlabel('Volume [km³]',fontsize=18)
plt.legend((p2[0], p4[0], p1[0], fari[0]),
           ('Volume without frontal ablation',
            'Volume with frontal ablation',
            'Volume below sea level',
            'Farinotti et al. (2019)'),
            frameon=True,
            bbox_to_anchor=(1.1, -0.15), ncol=3, fontsize=15)
plt.margins(0.05)

#plt.show()
plt.savefig(os.path.join(plot_path, 'marine_volume_draft.pdf'),
            dpi=150, bbox_inches='tight')