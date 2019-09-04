# This script will plot the results of calving vs volume experiment
import numpy as np
import pandas as pd
import os
import glob
import matplotlib.pyplot as plt
from matplotlib import rcParams
from collections import OrderedDict
import seaborn as sns
from mpl_toolkits.axes_grid1 import ImageGrid
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.offsetbox import AnchoredText

def get_slope(calving_values, volume_values):

    see = np.diff(percent)
    indexes = [i for i, e in enumerate(see) if e != 0]
    m, b = np.polyfit(calving_values, volume_values, 1)
    return m

def percentage(list):
    percent = list / list[0]
    return percent

MAIN_PATH = os.path.expanduser('~/cryo_calving_2019/')
WORKING_DIR = os.path.join(MAIN_PATH,
    'output_data/2_Calving_vs_Volume_exp/sensitivity_calvsvolRGI60-01.*.csv')
plot_path = os.path.join(MAIN_PATH, 'plots/')

#Sort files
filenames = sorted(glob.glob(WORKING_DIR))

path_climate_file = os.path.join(MAIN_PATH,
   'output_data/2_Calving_vs_Volume_exp/precipitation/precipitation_calving_glaciers.csv')

climate = pd.read_csv(path_climate_file)
climate = climate.sort_values('RGI_ID',
                              ascending = True).reset_index(drop = True)

# Plot settings
rcParams['axes.labelsize'] = 20
rcParams['xtick.labelsize'] = 20
rcParams['ytick.labelsize'] = 20
rcParams['legend.fontsize'] = 18
# Set figure width and height in cm
width_cm = 12
height_cm = 7

fig3 = plt.figure(3, figsize=(width_cm, height_cm*2))

llkw = {'interval': 0}
letkm = dict(color='black', ha='left', va='top', fontsize=20,
             bbox=dict(facecolor='white', edgecolor='black'))

percent = []

my_labels = {"x1": "volume > 500 km³", "x2": "500 km³ > volume > 100 km³",
             "x3": "100 km³ > volume > 10 km³", "x4": "10 km³ > volume > 0 km³"}

ax1 = plt.subplot(211)
sns.set_style("white")
ax1.tick_params(axis='both', bottom=True,
               left=True, width=2, direction='out', length=5)

for j, f in enumerate(filenames):
    #Get each glacier file and variables
    glacier = pd.read_csv(f)
    calving = glacier['calving_flux']
    volume = glacier['volume']

    #Get climate variable
    prcp = climate['precp'][j]

    if volume[0] > 500:
        percent = volume / volume[0]
        see = np.diff(percent)
        indexes = [i for i, e in enumerate(see) if e != 0]
        plt.plot(calving/prcp, percent, sns.xkcd_rgb["burnt orange"],
                 label=my_labels["x1"], linewidth=2.5)
    if 500 > volume[0] > 100:
        percent = volume / volume[0]
        # Eliminate constant values
        see = np.diff(percent)
        indexes = [i for i, e in enumerate(see) if e!=0]
        plt.plot(calving[indexes]/prcp, percent[indexes], sns.xkcd_rgb["teal green"],
                     label=my_labels["x2"], linewidth=2.5, alpha=0.5)
    if 100 > volume[0] > 10:
        percent = volume / volume[0]
        # Eliminate constant values
        see = np.diff(percent)
        indexes = [i for i, e in enumerate(see) if e != 0]
        plt.plot(calving[indexes]/prcp, percent[indexes], sns.xkcd_rgb["ocean blue"],
                 label=my_labels["x3"], linewidth=2.5, alpha=0.3)
    # if  10 > volume[0] > 0:
    #     percent = volume / volume[0]
    #     see = np.diff(percent)
    #     indexes = [i for i, e in enumerate(see) if e != 0]
    #     plt.plot(calving[indexes], percent[indexes], sns.xkcd_rgb['pink'],
    #             label=my_labels["x4"],  linewidth=2.0)
    else:
        pass
    plt.ylabel('Normalised glacier volume', size=20)
    at = AnchoredText('a', prop=dict(size=20), frameon=True, loc=2)
    ax1.add_artist(at)
    plt.margins(0.05)

ax2 = plt.subplot(212)
ax2.tick_params(axis='both',
                bottom=True, left=True, width=2, direction='out', length=5)
sns.set_style("white")
for i, f in enumerate(filenames):
    # Get each glacier file and variables
    glacier = pd.read_csv(f)
    calving = glacier['calving_flux']
    mu_star = glacier['mu_star']
    volume = glacier['volume']

    # Get climate variable
    prcp = climate['precp'][i]

    if volume[0] > 500:
        percent = volume / volume[0]
        see = np.diff(percent)
        indexes = [i for i, e in enumerate(see) if e != 0]
        plt.plot(calving/prcp, mu_star, sns.xkcd_rgb["burnt orange"],
                 label=my_labels["x1"], linewidth=2.5)
    if 500 > volume[0] > 100:
        percent = volume / volume[0]
        # Eliminate constant values
        see = np.diff(percent)
        indexes = [i for i, e in enumerate(see) if e != 0]
        plt.plot(calving[indexes]/prcp, mu_star[indexes], sns.xkcd_rgb["teal green"],
                 label=my_labels["x2"], linewidth=2.5, alpha=0.5)
    if 100 > volume[0] > 10:
        percent = volume / volume[0]
        # Eliminate constant values
        see = np.diff(percent)
        indexes = [i for i, e in enumerate(see) if e != 0]
        plt.plot(calving[indexes]/prcp, mu_star[indexes], sns.xkcd_rgb["ocean blue"],
                  label=my_labels["x3"], linewidth=2.5, alpha=0.5)
    # if  10 > volume[0] > 0:
    #     plt.plot(calving[0:150], mu_star[0:150], sns.xkcd_rgb['pink'],
    #             label=my_labels["x4"],  linewidth=2.0)
    else:
        pass
    plt.xlabel('Frontal ablation fraction', size=20)
    plt.ylabel('Temperature sensitivity \n $\mu^{*}$ [mm $yr^{-1}K^{-1}$]', size=20)
    at = AnchoredText('b', prop=dict(size=20), frameon=True, loc=2)
    ax2.add_artist(at)
    #plt.text(-0.2, 153, 'b', **letkm)
    plt.margins(0.05)

handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys(), loc='upper right')

plt.show()
#plt.savefig(os.path.join(plot_path,'calving_volume_mu_draft.pdf'),
#                               dpi=150, bbox_inches='tight')

initial_volume = []
slopes = []

for j, f in enumerate(filenames):


    #Get each glacier file and variables
    glacier = pd.read_csv(f)
    calving = glacier['calving_flux']
    volume = glacier['volume']

    #Get climate variable
    prcp = climate['precp'][j]

    calving_values = calving/prcp
    percent = volume / volume[0]

    see = np.diff(percent)
    indexes = [i for i, e in enumerate(see) if e != 0]

    if len(indexes)<2:
        pass
    else:
        slope = get_slope(calving_values[indexes], percent[indexes])

        initial_volume = np.append(initial_volume, volume[0])
        slopes = np.append(slopes, slope)


fig4 = plt.figure(1, figsize=(width_cm, height_cm))
plt.scatter(initial_volume, slopes)
plt.xlabel('Volume', size=20)
plt.ylabel('Sensitivity to calving', size=20)
plt.show()