# This script will plot the results of kxfactors experiment
import numpy as np
import pandas as pd
import os
import glob
import seaborn as sns
os.getcwd()
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.offsetbox import AnchoredText

# Functions that we need
def read_experiment_file(filename):
    glacier = pd.read_csv(filename)
    glacier = glacier[['rgi_id','terminus_type', 'calving_flux']]#, 'mu_star']]
    calving = glacier['calving_flux']
    return calving

# State main directory path
MAIN_PATH = os.path.expanduser('~/cryo_calving_2019/')
plot_path = os.path.join(MAIN_PATH, 'plots/')


############################################# Reading k x factors experiments
EXPERIMENT_PATH = os.path.join(MAIN_PATH,
                        'output_data/3_Sensitivity_studies_k_A_fs/3_1_k_exp/')

#K x factors directories
WORKING_DIR_one= os.path.join(EXPERIMENT_PATH,
         '1_k_parameter_exp/glacier_*.csv') #1
WORKING_DIR_two = os.path.join(EXPERIMENT_PATH,
        '2_k_parameter_exp/glacier_*.csv')  #2

filenames = []
filenames.append(sorted(glob.glob(WORKING_DIR_one)))
filenames.append(sorted(glob.glob(WORKING_DIR_two)))

kfactors = np.arange(0.10,1.10,0.02)
k = np.asarray(kfactors)*2.4

calvings = []
for elem in filenames:
    res = []
    for f in elem:
        res.append(read_experiment_file(f))
    calvings.append(res)

calving_fluxes = []
for cf in calvings:
    calving_fluxes.append(pd.concat([c for c in cf], axis=1))
    calving_fluxes[-1] = calving_fluxes[-1][(calving_fluxes[-1].T != 0).any()]
    calving_fluxes[-1] = calving_fluxes[-1].reset_index(drop=True)

cf1 = calving_fluxes[0]
cf2 = calving_fluxes[1]

cf1 = cf1['calving_flux'].sum(axis=0)
cf2 = cf2['calving_flux'].sum(axis=0)

data_frame1 = cf1.T
data_frame2 = cf2.T

#Observations
#Observations
lw_bound = np.repeat(15.11 * 1.091-3.96, len(k))
obs = np.repeat(15.11 * 1.091, len(k))
up_bound= np.repeat(15.11 * 1.091+3.96, len(k))

#From the find_intercepts scripts we copy
m3 = 0.0
b3 = 16.48501
k1 = 0.63208
k2 = 0.6659
k1_low = 0.50369
k2_up = 0.81968

my_labels_k = {"x1": "A = OGGM default, fs = 0.0",
               "x2": "A = OGGM default, fs = OGGM default"}

######################################### Reading glen a x factors experiments
EXPERIMENT_PATH_A = os.path.join(MAIN_PATH,
                'output_data/3_Sensitivity_studies_k_A_fs/3_2_glen_A_exp/')

#Glen a x factor directories
WORKING_DIR_zero_A = os.path.join(EXPERIMENT_PATH_A,
                                'glena_exp1/glacier_*.csv')     #0
WORKING_DIR_one_A = os.path.join(EXPERIMENT_PATH_A,
                                'glena_exp2/glacier_*.csv')    #1
WORKING_DIR_two_A = os.path.join(EXPERIMENT_PATH_A,
                                'glena_exp3/glacier_*.csv')      #2
WORKING_DIR_three_A = os.path.join(EXPERIMENT_PATH_A,
                                'glena_exp4/glacier_*.csv')      #3

filenames_A = []
filenames_A.append(sorted(glob.glob(WORKING_DIR_zero_A)))
filenames_A.append(sorted(glob.glob(WORKING_DIR_one_A)))
filenames_A.append(sorted(glob.glob(WORKING_DIR_two_A)))
filenames_A.append(sorted(glob.glob(WORKING_DIR_three_A)))

# Glen_a array
factors = np.arange(0.5,2.0,0.02)
glen_a = np.asarray(factors)*2.4e-24

calvings_A = []
for elem in filenames_A:
    res = []
    for f in elem:
        res.append(read_experiment_file(f))
    calvings_A.append(res)

calving_fluxes_A = []

for cf in calvings_A:
    calving_fluxes_A.append(pd.concat([c for c in cf], axis=1))
    calving_fluxes_A[-1] = calving_fluxes_A[-1][(calving_fluxes_A[-1].T != 0).any()]
    calving_fluxes_A[-1] = calving_fluxes_A[-1].reset_index(drop=True)


cf0_A = calving_fluxes_A[0]
cf1_A = calving_fluxes_A[1]

cf2_A = calving_fluxes_A[2]
cf3_A = calving_fluxes_A[3]

cf0_A = cf0_A['calving_flux'].sum(axis=0)
cf1_A = cf1_A['calving_flux'].sum(axis=0)

cf2_A = cf2_A['calving_flux'].sum(axis=0)
cf3_A = cf3_A['calving_flux'].sum(axis=0)

data_frame0_A = cf0_A.T
data_frame1_A = cf1_A.T

data_frame2_A = cf2_A.T
data_frame3_A = cf3_A.T

#From find_intercepts.py we copy
glen_0 = 2.4057015372011198e-24
glen_1 = 2.70310265604804e-24
glen_2 = 2.1148709491675595e-24
glen_3 = 2.4018556703316452e-24

glena_low = 4.6710383205133994e-24
glena_up = 1.2954387147693303e-24

my_labels_glena = {"x0": "fs = 0.0, " + 'k1 = '+ str(round(k1, 2)),
                   "x1": "fs = 0.0, " + 'k2 = '+ str(round(k2, 2)),
                   "x2": "fs = OGGM default, " + 'k1 = ' + str(round(k1, 2)),
                   "x3": "fs = OGGM default, " + 'k2 = ' + str(round(k2, 2))}

######################################### Reading fs x factors experiments

EXPERIMENT_PATH_fs = os.path.join(MAIN_PATH,
                'output_data/3_Sensitivity_studies_k_A_fs/3_3_fs_exp/')

WORKING_DIR_one_fs = os.path.join(EXPERIMENT_PATH_fs, 'fs_exp1/glacier_*.csv') #1

WORKING_DIR_two_fs = os.path.join(EXPERIMENT_PATH_fs, 'fs_exp2/glacier_*.csv')#2

filenames_fs = []
filenames_fs.append(sorted(glob.glob(WORKING_DIR_one_fs)))
filenames_fs.append(sorted(glob.glob(WORKING_DIR_two_fs)))

# Glen_a array or FS
fsfactors = np.arange(0.00,50.00,5)
fs = np.asarray(fsfactors)*5.7e-20

calvings_fs = []

for elem in filenames_fs:
    res = []
    for f in elem:
        res.append(read_experiment_file(f))
    calvings_fs.append(res)

calving_fluxes_fs = []

for cf in calvings_fs:
    calving_fluxes_fs.append(pd.concat([c for c in cf], axis=1))
    calving_fluxes_fs[-1] = calving_fluxes_fs[-1][(calving_fluxes_fs[-1].T != 0).any()]
    calving_fluxes_fs[-1] = calving_fluxes_fs[-1].reset_index(drop=True)

cf1_fs = calving_fluxes_fs[0]
cf2_fs = calving_fluxes_fs[1]

cf1_fs = cf1_fs['calving_flux'].sum(axis=0)
cf2_fs = cf2_fs['calving_flux'].sum(axis=0)

data_frame1_fs = cf1_fs.T
data_frame2_fs = cf2_fs.T

my_labels_fs = {"x1": "A = OGGM default, " + 'k1 = '+ str(round(k1, 2)),
                "x2": "A = OGGM default, " 'k2 = '+ str(round(k2, 2))}

fs_low = 2.598522451128765e-19
######################################## plot ######################################
# Plot settings
rcParams['axes.labelsize'] = 20
rcParams['xtick.labelsize'] = 20
rcParams['ytick.labelsize'] = 20
rcParams['legend.fontsize'] = 20
# Set figure width and height in cm
width_cm = 14
height_cm = 7

from matplotlib import gridspec

# Create figure and axes instances
fig1 = plt.figure(1, figsize=(width_cm, height_cm*3))

gs = gridspec.GridSpec(3, 1, hspace=0.2)

#sns.set_color_codes("colorblind")
ax1 = plt.subplot(gs[0])
sns.set_style("white")
ax1.tick_params(axis='both', bottom=True, left=True, width=2, direction='out', length=5)
plt.plot(k, data_frame1, "o", color=sns.xkcd_rgb["ocean blue"],
             linewidth=2.5, markersize=10,
             label=my_labels_k["x1"])

plt.plot(k, data_frame2, "o", color=sns.xkcd_rgb["teal green"],
             linewidth=2.5, markersize=10,
             label=my_labels_k["x2"])

plt.plot(k1, obs[0], 'x', markersize=12,
         color=sns.xkcd_rgb["ocean blue"], linewidth=4,
         label='k1 = '+ str(round(k1, 2)))
plt.plot(k2, obs[0], 'x', markersize=12,
         color=sns.xkcd_rgb["teal green"], linewidth=4,
         label='k2 = '+ str(round(k2, 2)))

plt.plot(k1_low, lw_bound[0], 'x', markersize=12,
         color=sns.xkcd_rgb["black"], linewidth=5)
       # label='k1 = '+ str(round(k1, 2))

plt.plot(k2_up, up_bound[0], 'x', markersize=12,
         color=sns.xkcd_rgb["black"], linewidth=5)

plt.plot(k, b3 + m3*k, '--', color='black', linewidth=3.0,
        label='Frontal ablation (McNabb et al., 2015)')
plt.fill_between(k, (b3 + m3*k) - 3.96, (b3 + m3*k) + 3.96,
                 color=sns.xkcd_rgb["grey"], alpha=0.3)

plt.gca().axes.get_xaxis().set_visible(True)
plt.ylabel('Alaska frontal ablation \n [$km³.yr^{-1}$]')
plt.xlabel('Calving constant k [$\mathregular{yr^{-1}}$] ')
plt.legend(loc='lower right', borderaxespad=0.)
at = AnchoredText('a', prop=dict(size=20), frameon=True, loc=2)
ax1.add_artist(at)

plt.margins(0.05)


ax2 = plt.subplot(gs[1])
ax2.tick_params(axis='both', bottom=True, left=True, width=2, direction='out', length=5)
sns.set_color_codes("colorblind")
sns.set_style("white")

plt.plot(glen_a, data_frame0_A, linestyle="--",
             label=my_labels_glena["x0"], linewidth=2.5)

plt.plot(glen_a, data_frame1_A, linestyle="--",
             label=my_labels_glena["x1"], linewidth=2.5)

plt.plot(glen_a, data_frame2_A,
             label=my_labels_glena["x2"], linewidth=2.5)

plt.plot(glen_a, data_frame3_A,
             label=my_labels_glena["x3"], linewidth=2.5)


plt.plot(glen_0, obs[0], 'x',
         markersize=20, linewidth=4, color=sns.xkcd_rgb["ocean blue"])

plt.plot(glen_1, obs[0], 'x',
         markersize=20, linewidth=4, color=sns.xkcd_rgb["orange"])

plt.plot(glen_2, obs[0], 'x',
         markersize=20, linewidth=4, color=sns.xkcd_rgb["green"])

plt.plot(glen_3, obs[0], 'x',
         markersize=20, linewidth=4, color=sns.xkcd_rgb["red"])

plt.plot(glena_low, lw_bound[0], 'x',
         markersize=20, linewidth=5, color=sns.xkcd_rgb["black"])

plt.plot(glena_up, up_bound[0], 'x',
         markersize=20, linewidth=5, color=sns.xkcd_rgb["black"])

plt.plot(glen_a, np.repeat(15.11*1.091, len(glen_a)), '--', color='black',
             label='Frontal ablation (McNabb et al., 2015)', linewidth=3.0)

plt.fill_between(glen_a,np.repeat(15.11*1.091-3.96, len(glen_a)),
                 np.repeat(15.11*1.091+3.96, len(glen_a)),
                 color=sns.xkcd_rgb["grey"], alpha=0.3)

plt.gca().axes.get_xaxis().set_visible(True)
plt.ylabel('Alaska frontal ablation \n [$km³.yr^{-1}$]')
plt.xlabel('Glen A [$\mathregular{s^{−1}} \mathregular{Pa^{−3}}$]')

plt.legend(loc='upper right', borderaxespad=0.)

at = AnchoredText('b', prop=dict(size=20), frameon=True, loc=2)
ax2.add_artist(at)

plt.margins(0.05)


ax3 = plt.subplot(gs[2])
ax3.tick_params(axis='both', bottom=True, left=True, width=2, direction='out', length=5)
sns.set_color_codes("colorblind")
sns.set_style("white")

plt.plot(fs[:-2], data_frame1_fs[:-2],
         label=my_labels_fs["x1"], linewidth=2.5)

plt.plot(fs[:-2], data_frame2_fs[:-2],
         label=my_labels_fs["x2"], linewidth=2.5)

plt.plot(fs[:-2], np.repeat(15.11 * 1.091, len(fs[:-2])), '--', color='black',
         label='Frontal ablation (McNabb et al., 2015)', linewidth=3.0)
#
plt.plot(fs_low, lw_bound[0], 'x',
         markersize=20, linewidth=5, color=sns.xkcd_rgb["black"])

plt.fill_between(fs[:-2],np.repeat(15.11*1.091-3.96, len(fs[:-2])),
                 np.repeat(15.11*1.091+3.96, len(fs[:-2])),
                 color=sns.xkcd_rgb["grey"], alpha=0.3)

plt.gca().axes.get_xaxis().set_visible(True)

plt.ylabel('Alaska frontal ablation \n [$km³.yr^{-1}$]')
plt.xlabel('Sliding parameter $f_{s}$ [$\mathregular{s^{−1}} \mathregular{Pa^{−3}}$]')
plt.legend(loc='upper right', borderaxespad=0.)

at = AnchoredText('c', prop=dict(size=20), frameon=True, loc=2)
ax3.add_artist(at)

plt.margins(0.05)

plt.subplots_adjust(hspace=0.2)
plt.tight_layout()
#plt.show()

plt.savefig(os.path.join(plot_path, 'sensitivity_draft.pdf'), dpi=150,
                      bbox_inches='tight')
