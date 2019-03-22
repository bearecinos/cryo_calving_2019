# This script will plot the workflow for the columbia glacier, the calving output
# and the columbia bed profile along the main centreline


"""Useful plotting functions"""
import netCDF4
import pandas as pd
import shapely.geometry as shpg
from matplotlib import rcParams
from scipy import stats
import os
import geopandas as gpd
import numpy as np
import salem
import seaborn as sns

from oggm import cfg, graphics

import oggm.utils
import matplotlib.pyplot as plt
import tarfile

cfg.initialize()
MAIN_PATH = os.path.expanduser('~/Documents/k_experiments/')

WORKING_DIR = os.path.join(MAIN_PATH,
                           'output_data/1_Columbia_Glacier_runs/2_calving/')

RGI_FILE = os.path.join(WORKING_DIR,
                'per_glacier/RGI60-01/RGI60-01.10/RGI60-01.10689/outlines.shp')

cfg.PATHS['dem_file'] = os.path.join(WORKING_DIR,
                    'per_glacier/RGI60-01/RGI60-01.10/RGI60-01.10689/dem.tif')

entity = gpd.read_file(RGI_FILE).iloc[0]

gdir = oggm.GlacierDirectory(entity,
                             base_dir=os.path.join(WORKING_DIR, 'per_glacier'))

plot_path = os.path.join(MAIN_PATH, 'plots/')

k = 2.4


# Visualizing calving output
# # ----------------

cl = gdir.read_pickle('calving_output', filesuffix='_old')
cl_new = gdir.read_pickle('calving_output')

F_a = []
mu_star = []

F_a_new = []
mu_star_new = []

for c, c_new in zip(cl,cl_new):
    F_a.extend(c['calving_fluxes'])
    mu_star.extend(c['mu_star_calving'])
    F_a_new.extend(c_new['calving_fluxes'])
    mu_star_new.extend(c_new['mu_star_calving'])

iteration = np.arange(0,len(F_a),1)

iteration_new = np.arange(0,len(F_a_new),1)

print(iteration)
print(iteration_new)
print(F_a_new)
print(mu_star_new)


rcParams['axes.labelsize'] = 20
rcParams['xtick.labelsize'] = 20
rcParams['ytick.labelsize'] = 20
rcParams['legend.fontsize'] = 18

def two_scales(ax1, time, data1, data2, c1, c2):
    ax2 = ax1.twinx()
    ax1.plot(time, data1, c1)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Frontal ablation \n [km³$yr^{-1}$]', color='b')
    ax2.plot(time, data2, c2)
    ax2.set_ylabel('Temperature sensitivity \n $\mu^{*}$ [mm $yr^{-1}K^{-1}$]', color='r')
    return ax1, ax2



fig, (ax1, ax2) = plt.subplots(1,2, figsize=(10,4))
sns.set()
sns.set_color_codes("colorblind")
sns.set(style="white", context="talk")

ax1, ax1a = two_scales(ax1, iteration, F_a, mu_star, 'b-', 'r-')

ax2, ax2a = two_scales(ax2, iteration_new, F_a_new, mu_star_new, 'b-', 'r-')

# Change color of each axis
def color_y_axis(ax, color):
    """Color your axes."""
    for t in ax.get_yticklabels():
        t.set_color(color)

color_y_axis(ax1, color='b')
color_y_axis(ax1a, color='r')
color_y_axis(ax2, color='b')
color_y_axis(ax2a, color='r')

fig.tight_layout()
#plt.show()
plt.savefig(os.path.join(plot_path, 'columbia_loop.pdf'),
                         dpi=150, bbox_inches='tight')


#print(F_a, mu_star)

