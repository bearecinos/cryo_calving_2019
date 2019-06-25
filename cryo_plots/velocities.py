import pandas as pd
import os
import geopandas as gpd
import numpy as np
import seaborn as sns
from oggm import cfg, utils
from oggm import workflow

import matplotlib.pyplot as plt
from matplotlib import rcParams

# Plot settings
rcParams['axes.labelsize'] = 20
rcParams['xtick.labelsize'] = 20
rcParams['ytick.labelsize'] = 20
rcParams['legend.fontsize'] = 16
# Set figure width and height in cm
width_cm = 12
height_cm = 8


# Reading glacier directories per experiment
MAIN_PATH = os.path.expanduser('~/cryo_calving_2019/')

plot_path = os.path.join(MAIN_PATH, 'plots/')

no_calving = os.path.join(MAIN_PATH,
'output_data/6_Velocities/6_Velocities/no_calving/')

with_calving = os.path.join(MAIN_PATH,
'output_data/6_Velocities/6_Velocities/calving/')

# Reading RGI
RGI_FILE = os.path.join(MAIN_PATH,
                'input_data/01_rgi60_Alaska_modify/01_rgi60_Alaska.shp')

def normalize(vel):
    vel_min = min(vel)
    vel_max = max(vel)

    n_vel = (vel - vel_min) / (vel_max - vel_min)
    return n_vel

def init_velocity(workdir):
    cfg.initialize()
    cfg.PARAMS['use_multiprocessing'] = False
    cfg.PATHS['working_dir'] = workdir
    cfg.PARAMS['border'] = 20

    # Read RGI file
    rgidf = gpd.read_file(RGI_FILE)

    # Run only for Marine terminating
    glac_type = [0, 2]
    keep_glactype = [(i not in glac_type) for i in rgidf.TermType]
    rgidf = rgidf.iloc[keep_glactype]

    # columbia = ['RGI60-01.10689']
    # keep_indexes = [(i in columbia) for i in rgidf.RGIId]
    # rgidf = rgidf.iloc[keep_indexes]

    return workflow.init_glacier_regions(rgidf)

def calculate_velocity(gdir):
    map_dx = gdir.grid.dx
    fls = gdir.read_pickle('inversion_flowlines')

    cl = gdir.read_pickle('inversion_input')[-1]
    inv = gdir.read_pickle('inversion_output')[-1]

    # vol in m3 and dx in m
    section = (inv['volume'] / inv['dx']) / 1e6

    x = np.arange(fls[-1].nx) * fls[-1].dx * map_dx * 1e-3

    # this flux is in m3 per second needs to be km3/sec
    flux = inv['flux'] / 1e9
    #print(flux)

    angle = cl['slope_angle']

    velocity = flux / section

    velocity *= cfg.SEC_IN_YEAR

    return velocity, normalize(x), angle

gdirs_one = init_velocity(no_calving)

gdirs_two = init_velocity(with_calving)

fig1 = plt.figure(1, figsize=(width_cm, height_cm))
sns.set_color_codes("colorblind")

for gdir, gdir_c in zip(gdirs_one,gdirs_two):
    vel, x, angle = calculate_velocity(gdir)
    vel_c, x_c, angle_c= calculate_velocity(gdir_c)

    #print(gdir.rgi_id)
    #print(vel_c)

    diff_angle_c = np.arctan(angle_c)
    diff_angle = np.diff(np.arctan(angle))

    if vel_c[-1] > 0.0:
        #print(vel_c)
        plt.plot(x_c, vel_c-vel, '-', label=gdir.rgi_id, linewidth=2.5)
        plt.legend(loc='upper left', ncol=2, bbox_to_anchor=(1.0, 1.02))
        plt.xlabel('Normalised distance along the main flowline')
        plt.ylabel('Velocity difference [$km.yr^{-1}$]')

plt.margins(0.05)
#plt.show()
plt.savefig(os.path.join(plot_path,'velocity_differences.pdf'),
                               dpi=150, bbox_inches='tight')
