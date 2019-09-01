# This script will find the intercepts for kxfactors experiment
# With the F_a flux calculated b y McNabb (2015)

import numpy as np
import pandas as pd
import os
import glob
import seaborn as sns
os.getcwd()
import matplotlib.pyplot as plt
from matplotlib import rcParams
from decimal import Decimal
import oggm.cfg as cfg
cfg.initialize()

#########  Plotting things
rcParams['axes.labelsize'] = 20
rcParams['xtick.labelsize'] = 20
rcParams['ytick.labelsize'] = 20
rcParams['legend.fontsize'] = 18
# Set figure width and height in cm
width_cm = 14
height_cm = 7

# Functions that we need
def read_experiment_file(filename):
    glacier = pd.read_csv(filename)
    glacier = glacier[['rgi_id','terminus_type', 'calving_flux']]#, 'mu_star']]
    calving = glacier['calving_flux']
    return calving

def getSlope(x1, y1, x2, y2):
    return (y2-y1)/(x2-x1)

def getYInt(x1, y1, x2, y2):
    slope = getSlope(x1, y1, x2, y2)
    y = y1-slope*x1
    return y

def get_lines_Int(m1,b1,m2,b2):
    x = (b2 - b1)/(m1 - m2)
    return x

def find_lower_uper_index(df, do):
    low = df[df['calving_flux'] < do[0]].index.values
    up = df[df['calving_flux'] > do[0]].index.values
    return low, up

def defining_points_k(lower, bigger, dk, df):
    """
    Gives back the points for k and
    the data_frame that we need to linearly interpolate
    lower: array with the index list of lower values
    bigger: array with the index list of bigger values
    k: k array
    df: data_frame of frontal ablation
    """

    points_k = np.asarray([dk[lower[-1]],
                           df.iloc[lower[-1]]['calving_flux'],
                           dk[bigger[0]],
                           df.iloc[bigger[0]]['calving_flux']])
    return points_k

def defining_points_A(lower, bigger, dA, df):
    """
        Gives back the points for glen A and
        the data_frame that we need to linearly interpolate
        lower: array with the index list of lower values
        bigger: array with the index list of bigger values
        dA: A array
        df: data_frame of frontal ablation
        """
    points_A = np.asarray([dA[bigger[-1]],
                           df.iloc[bigger[-1]]['calving_flux'],
                           dA[lower[0]],
                           df.iloc[lower[0]]['calving_flux']])
    return points_A



def find_slope_intercept(array):
    slope = getSlope(array[0], array[1], array[2], array[3])
    intercept = getYInt(array[0], array[1], array[2], array[3])
    return slope, intercept

# SOME GLOBAL SETTINGS
PLOT_K = False
PLOT_A = True
PLOT_FS = True

# Defining directories for data and plotting
MAIN_PATH = os.path.expanduser('~/cryo_calving_2019/')
plot_path = os.path.join(MAIN_PATH, 'plots/')
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

# Reading and organizing the data in dataframes
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

cf1 = cf1.reset_index(name='calving_flux')
cf2 = cf2.reset_index(name='calving_flux')
cf1 = cf1.drop('index', 1)
cf2 = cf2.drop('index', 1)

data_frame1 = cf1
data_frame2 = cf2

#Observations
lw_bound = np.repeat(15.11 * 1.091-3.96, len(k))
obs = np.repeat(15.11 * 1.091, len(k))
up_bound= np.repeat(15.11 * 1.091+3.96, len(k))

###### Find index below and above observations
# k1
k1_index_l, k1_index_u = find_lower_uper_index(data_frame1, obs)
# k2
k2_index_l, k2_index_u = find_lower_uper_index(data_frame2, obs)

###### Find index below and above uncertainty
# k1_lower
k1_lower_index_l, k1_lower_index_u = find_lower_uper_index(data_frame1,
                                                           lw_bound)
# k1_lower
k2_lower_index_l, k2_lower_index_u = find_lower_uper_index(data_frame2,
                                                           lw_bound)
# k1_upper
k1_upper_index_l, k1_upper_index_u = find_lower_uper_index(data_frame1,
                                                           up_bound)
# k2_upper
k2_upper_index_l, k2_upper_index_u = find_lower_uper_index(data_frame2,
                                                           up_bound)

#### Defining the points
points_k1 = defining_points_k(k1_index_l, k1_index_u, k, data_frame1)

points_k2 = defining_points_k(k2_index_l, k2_index_u, k, data_frame2)

# Uncertainty
points_k1_lw = defining_points_k(k1_lower_index_l,
                                  k1_lower_index_u,
                                  k, data_frame1)

points_k2_lw = defining_points_k(k2_lower_index_l,
                                  k2_lower_index_u,
                                  k,
                                  data_frame2)

points_k1_up = defining_points_k(k1_upper_index_l,
                                 k1_upper_index_u,
                                 k, data_frame1)

points_k2_up = defining_points_k(k2_upper_index_l,
                                 k2_upper_index_u,
                                 k,
                                 data_frame2)

# observations
points_obs = np.asanyarray([k[0], obs[0], k[1], obs[1]])
# Uncertainty
points_lw_bound = np.asanyarray([k[0], lw_bound[0], k[1], lw_bound[1]])
points_up_bound = np.asanyarray([k[0], up_bound[0], k[1], up_bound[1]])

# Finding slope and intercept for the lines between the points defined above
# for k1
m1, b1 = find_slope_intercept(points_k1)
# for k2
m2, b2 = find_slope_intercept(points_k2)

# for uncertainty limits
# for k1 low
m1_low , b1_low = find_slope_intercept(points_k1_lw)
# for k2 low
m2_low, b2_low = find_slope_intercept(points_k2_lw)
# for k1 up
m1_up , b1_up = find_slope_intercept(points_k1_up)
# for k2 up
m2_up, b2_up = find_slope_intercept(points_k2_up)

#for observations
m3, b3 = find_slope_intercept(points_obs)

#For uncetainty limits
m3_low, b3_low = find_slope_intercept(points_lw_bound)
m3_up, b3_up = find_slope_intercept(points_up_bound)

# Get the intercepts
k1 = np.around(get_lines_Int(m1, b1, m3, b3), 5)
k2 = np.around(get_lines_Int(m2, b2, m3, b3), 5)

k1_low = np.around(get_lines_Int(m1_low, b1_low, m3_low, b3_low), 5)
k2_low = np.around(get_lines_Int(m2_low, b2_low, m3_low, b3_low), 5)

k1_up = np.around(get_lines_Int(m1_up, b1_up, m3_up, b3_up), 5)
k2_up = np.around(get_lines_Int(m2_up, b2_up, m3_up, b3_up), 5)

print('k1', k1)
print('k2', k2)
print('k1_low', k1_low)
print('k2_up', k2_up)

if PLOT_K:
    my_labels_k = {"x1": "A = OGGM default, fs = 0.0",
                   "x2": "A = OGGM default, fs = OGGM default"}

    fig = plt.figure(1, figsize=(width_cm, height_cm))
    sns.set_color_codes("colorblind")
    sns.set_style("white")

    plt.plot(k, data_frame1, "o", color=sns.xkcd_rgb["ocean blue"],
             linewidth=2.5, markersize=12,
             label=my_labels_k["x1"])

    plt.plot(k, data_frame2, "o", color=sns.xkcd_rgb["teal green"],
             linewidth=2.5, markersize=12,
             label=my_labels_k["x2"])

    plt.plot(k, b3 + m3 * k, '--', color='black', linewidth=3.0,
             label='Frontal ablation (McNabb et al., 2015)')

    plt.plot(k1, obs[0], 'x', markersize=20,
             color=sns.xkcd_rgb["ocean blue"], linewidth=4,
             label='k1 =' + str(round(k1, 2)))

    plt.plot(k1_low, lw_bound[0], 'x', markersize=20,
             color=sns.xkcd_rgb["red"], linewidth=4,
             label='k1_low =' + str(round(k1_low, 2)))

    plt.plot(k2, obs[0], 'x', markersize=20,
             color=sns.xkcd_rgb["teal green"], linewidth=4,
             label='k2 =' + str(round(k2, 2)))

    plt.plot(k2_up, up_bound[0], 'x', markersize=20,
             color=sns.xkcd_rgb["red"], linewidth=4,
             label='k2_up =' + str(round(k2_up, 2)))

    plt.fill_between(k, (b3 + m3 * k) - 3.96, (b3 + m3 * k) + 3.96,
                     color=sns.xkcd_rgb["grey"], alpha=0.3)

    plt.gca().axes.get_xaxis().set_visible(True)

    plt.ylabel('Alaska frontal ablation \n [$km³.yr^{-1}$]')
    plt.xlabel('Calving constant k [$\mathregular{yr^{-1}}$] ')
    plt.legend(loc='lower right', borderaxespad=0.)

    letkm = dict(color='black', ha='left', va='top', fontsize=20,
                 bbox=dict(facecolor='white', edgecolor='black'))

    plt.margins(0.05)
    plt.show()
    # plt.savefig(os.path.join(plot_path, 'sensitivity_Alaska.pdf'), dpi=150,
    #                       bbox_inches='tight')

################### reading Glen A exp ##########################################

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

# print(filenames_A)

# Glen_a array or FS
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


cf0_A = cf0_A.reset_index(name='calving_flux')
cf1_A = cf1_A.reset_index(name='calving_flux')
cf2_A = cf2_A.reset_index(name='calving_flux')
cf3_A = cf3_A.reset_index(name='calving_flux')

cf0_A = cf0_A.drop('index', 1)
cf1_A = cf1_A.drop('index', 1)
cf2_A = cf2_A.drop('index', 1)
cf3_A = cf3_A.drop('index', 1)


data_frame0_A = cf0_A
data_frame1_A = cf1_A

data_frame2_A = cf2_A
data_frame3_A = cf3_A

###### Find index below and above observations
# glen_a0
A0_index_l, A0_index_u = find_lower_uper_index(data_frame0_A, obs)
# glen_a1
A1_index_l, A1_index_u = find_lower_uper_index(data_frame1_A, obs)
# glen_a2
A2_index_l, A2_index_u = find_lower_uper_index(data_frame2_A, obs)
# glen_a3
A3_index_l, A3_index_u = find_lower_uper_index(data_frame3_A, obs)

###### Find index below and above uncertainty
# glen_a0_lower
A0_lower_index_l, A0_lower_index_u = find_lower_uper_index(data_frame0_A,
                                                           lw_bound)
# glen_a1_lower
A1_lower_index_l, A1_lower_index_u = find_lower_uper_index(data_frame1_A,
                                                           lw_bound)
# glen_a2_lower
A2_lower_index_l, A2_lower_index_u = find_lower_uper_index(data_frame2_A,
                                                           lw_bound)
# glen_a3_lower
A3_lower_index_l, A3_lower_index_u = find_lower_uper_index(data_frame3_A,
                                                           lw_bound)
# glen_a0_upper
A0_upper_index_l, A0_upper_index_u = find_lower_uper_index(data_frame0_A,
                                                           up_bound)
# glen_a1_upper
A1_upper_index_l, A1_upper_index_u = find_lower_uper_index(data_frame1_A,
                                                           up_bound)
# glen_a2_upper
A2_upper_index_l, A2_upper_index_u = find_lower_uper_index(data_frame2_A,
                                                           up_bound)
# glen_a3_upper
A3_upper_index_l, A3_upper_index_u = find_lower_uper_index(data_frame3_A,
                                                           up_bound)

#### Defining the points that cross observations
points_glena_0 = defining_points_A(A0_index_l, A0_index_u, glen_a, data_frame0_A)
points_glena_1 = defining_points_A(A1_index_l, A1_index_u, glen_a, data_frame1_A)
points_glena_2 = defining_points_A(A2_index_l, A2_index_u, glen_a, data_frame2_A)
points_glena_3 = defining_points_A(A3_index_l, A3_index_u, glen_a, data_frame3_A)

## Defining the points that cross the uncertainty limits
# lOWER
points_glena0_lw = defining_points_A(A0_lower_index_l, A0_lower_index_u,
                                     glen_a, data_frame0_A)
points_glena1_lw = defining_points_A(A1_lower_index_l, A1_lower_index_u,
                                     glen_a, data_frame1_A)
points_glena2_lw = defining_points_A(A2_lower_index_l, A2_lower_index_u,
                                     glen_a, data_frame2_A)
points_glena3_lw = defining_points_A(A3_lower_index_l, A3_lower_index_u,
                                     glen_a, data_frame3_A)
# UPPER
points_glena0_up = defining_points_A(A0_upper_index_l, A0_upper_index_u,
                                     glen_a, data_frame0_A)
points_glena1_up = defining_points_A(A1_upper_index_l, A1_upper_index_u,
                                     glen_a, data_frame1_A)
points_glena2_up = defining_points_A(A2_upper_index_l, A2_upper_index_u,
                                     glen_a, data_frame2_A)
points_glena3_up = defining_points_A(A3_upper_index_l, A3_upper_index_u,
                                     glen_a, data_frame3_A)

# observations
points_obs_a = np.asanyarray([glen_a[0], obs[0], glen_a[1], obs[1]])
# Uncertainty
points_lw_bound_a = np.asanyarray([glen_a[0], lw_bound[0],
                                   glen_a[1], lw_bound[1]])

points_up_bound_a = np.asanyarray([glen_a[0], up_bound[0],
                                   glen_a[1], up_bound[1]])


# Finding slope and intercept for the lines between the points defined above
#glen_A's
m0_a, b0_a = find_slope_intercept(points_glena_0)
m1_a, b1_a = find_slope_intercept(points_glena_1)
m2_a, b2_a = find_slope_intercept(points_glena_2)
m3_a, b3_a = find_slope_intercept(points_glena_3)
# for observations
m4_obs, b4_obs = find_slope_intercept(points_obs_a)

# for uncertainty limits
# lower bound
m0_a_low, b0_a_low = find_slope_intercept(points_glena0_lw)
m1_a_low, b1_a_low = find_slope_intercept(points_glena1_lw)
m2_a_low, b2_a_low = find_slope_intercept(points_glena2_lw)
m3_a_low, b3_a_low = find_slope_intercept(points_glena3_lw)
# upper bound
m0_a_up, b0_a_up = find_slope_intercept(points_glena0_up)
m1_a_up, b1_a_up = find_slope_intercept(points_glena1_up)
m2_a_up, b2_a_up = find_slope_intercept(points_glena2_up)
m3_a_up, b3_a_up = find_slope_intercept(points_glena3_up)

# for uncertainty limits
m4_low, b4_low = find_slope_intercept(points_lw_bound_a)
m4_up, b4_up = find_slope_intercept(points_up_bound_a)

# Get intercepts
glena_0 = get_lines_Int(m0_a, b0_a, m4_obs, b4_obs)
glena_1 = get_lines_Int(m1_a, b1_a, m4_obs, b4_obs)
glena_2 = get_lines_Int(m2_a, b2_a, m4_obs, b4_obs)
glena_3 = get_lines_Int(m3_a, b3_a, m4_obs, b4_obs)

# with uppper uncertainty bound
glena_0_lw = get_lines_Int(m0_a_low, b0_a_low, m4_low, b4_low)
glena_1_lw = get_lines_Int(m1_a_low, b1_a_low, m4_low, b4_low)
glena_2_lw = get_lines_Int(m2_a_low, b2_a_low, m4_low, b4_low)
glena_3_lw = get_lines_Int(m3_a_low, b3_a_low, m4_low, b4_low)

# with the lower uncertainty bound
glena_0_up = get_lines_Int(m0_a_up, b0_a_up, m4_up, b4_up)
glena_1_up = get_lines_Int(m1_a_up, b1_a_up, m4_up, b4_up)
glena_2_up = get_lines_Int(m2_a_up, b2_a_up, m4_up, b4_up)
glena_3_up = get_lines_Int(m3_a_up, b3_a_up, m4_up, b4_up)

print('glen_0', glena_0)
print('glen_1', glena_1)
print('glen_2', glena_2)
print('glen_3', glena_3)

print('glena_low', glena_1_lw)
print('glena_up', glena_2_up)

###########################plot glen a experiments ################################
if PLOT_A:
    fig = plt.figure(2, figsize=(width_cm, height_cm))

    my_labels_glena = {"x0": "fs = 0.0, " + 'k1 = ' + str(round(k1, 2)),
                       "x1": "fs = 0.0, " + 'k2 = ' + str(round(k2, 2)),
                       "x2": "fs = OGGM default, " + 'k1 = ' + str(
                           round(k1, 2)),
                       "x3": "fs = OGGM default, " + 'k2 = ' + str(
                           round(k2, 2))}
    sns.set_color_codes("colorblind")
    sns.set_style("white")

    plt.plot(glen_a, data_frame0_A, linestyle="--",
             # color=sns.xkcd_rgb["forest green"],
             label=my_labels_glena["x0"], linewidth=2.5)

    plt.plot(glen_a, data_frame1_A, linestyle="--",
             # color=sns.xkcd_rgb["green"],
             label=my_labels_glena["x1"], linewidth=2.5)

    plt.plot(glen_a, data_frame2_A,  # color=sns.xkcd_rgb["teal"],
             label=my_labels_glena["x2"], linewidth=2.5)

    plt.plot(glen_a, data_frame3_A,  # color=sns.xkcd_rgb["turquoise"],
             label=my_labels_glena["x3"], linewidth=2.5)

    plt.plot(glena_0, obs[0], 'x', markersize=20, linewidth=4)

    plt.plot(glena_1, obs[0], 'x', markersize=20,
             linewidth=4)

    plt.plot(glena_2, obs[0], 'x', markersize=20,
             linewidth=4)

    plt.plot(glena_3, obs[0], 'x', markersize=20,
             linewidth=4)

    # lower bound intercepts
    plt.plot(glena_1_lw, lw_bound[0], 'x', markersize=20,
             linewidth=4, label='glen_1_lw')

    # upper bound intercepts
    plt.plot(glena_2_up, up_bound[0], 'x', markersize=20,
             linewidth=4, label='glen_2_up')

    plt.plot(glen_a, np.repeat(15.11 * 1.091, len(glen_a)), '--',
             color='black',
             label='Frontal ablation (McNabb et al., 2015)', linewidth=3.0)

    plt.fill_between(glen_a, np.repeat(15.11 * 1.091 - 3.96, len(glen_a)),
                     np.repeat(15.11 * 1.091 + 3.96, len(glen_a)),
                     color=sns.xkcd_rgb["grey"], alpha=0.3)

    # plt.xticks(glen_a)
    # plt.gca().axes.get_xticklines()
    # plt.yticks(np.arange(0, 35, 3.0))

    plt.ylabel('Alaska frontal ablation \n [$km³.yr^{-1}$]')
    plt.xlabel('Glen A [$\mathregular{s^{−1}} \mathregular{Pa^{−3}}$]')
    plt.legend(loc='upper right', borderaxespad=0.)
    letkm = dict(color='black', ha='left', va='top', fontsize=20,
                 bbox=dict(facecolor='white', edgecolor='black'))
    # plt.text(glen_a[0]-1.15e-25, 27.25, 'b', **letkm)

    plt.margins(0.05)
    plt.show()

############################## Reading fs x factors experiments ###############
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

data_frame1_fs = cf1_fs
data_frame2_fs = cf2_fs

data_frame1_fs = data_frame1_fs.reset_index()
data_frame2_fs = data_frame2_fs.reset_index()

data_frame1_fs = data_frame1_fs.drop(['index'], axis=1)
data_frame2_fs = data_frame2_fs.drop(['index'], axis=1)

data_frame1_fs = data_frame1_fs.rename(columns={data_frame1_fs.columns[0]: 'calving_flux'})
data_frame2_fs = data_frame2_fs.rename(columns={data_frame2_fs.columns[0]: 'calving_flux'})

my_labels_fs = {"x1": "A = OGGM default, " + 'k1 = '+ str(round(k1, 2)),
                "x2": "A = OGGM default, " 'k2 = '+ str(round(k2, 2))}


###### Find index below and above observations
# Uncertainty limits
# fs2_lower
fs_l, fs_u = find_lower_uper_index(data_frame2_fs[:-2], lw_bound[:-2])

# Get the points
points_fs = defining_points_A(fs_l, fs_u,
                              fs[:-2], data_frame2_fs[:-2])

# Uncertainty
points_lw_bound_fs = np.asanyarray([fs[0], lw_bound[0],
                                   fs[1], lw_bound[1]])

# Finding slope and intercept for the lines between the points defined above
#fs_2
m0_fs, b0_fs = find_slope_intercept(points_fs)

# for uncertainty limits
m1_fs_low, b1_fs_low = find_slope_intercept(points_lw_bound_fs)

# Get intercepts
fs_0 = get_lines_Int(m1_fs_low, b1_fs_low, m0_fs, b0_fs)

print('fs_low', fs_0)

if PLOT_FS:
    fig = plt.figure(2, figsize=(width_cm, height_cm))
    sns.set_color_codes("colorblind")
    sns.set_style("white")

    plt.plot(fs[:-2], data_frame1_fs[:-2],
             label=my_labels_fs["x1"], linewidth=2.5)

    plt.plot(fs[:-2], data_frame2_fs[:-2],
             label=my_labels_fs["x2"], linewidth=2.5)

    plt.plot(fs[:-2], np.repeat(15.11 * 1.091, len(fs[:-2])), '--',
             color='black',
             label='Frontal ablation (McNabb et al., 2015)', linewidth=3.0)

    plt.fill_between(fs[:-2], np.repeat(15.11 * 1.091 - 3.96, len(fs[:-2])),
                     np.repeat(15.11 * 1.091 + 3.96, len(fs[:-2])),
                     color=sns.xkcd_rgb["grey"], alpha=0.3)

    # lower bound intercepts
    plt.plot(fs_0, lw_bound[0], 'x', markersize=20,
             linewidth=4, label='fs_lw')

    plt.gca().axes.get_xaxis().set_visible(True)

    plt.ylabel('Alaska frontal ablation \n [$km³.yr^{-1}$]')
    plt.xlabel(
        'Sliding parameter $f_{s}$ [$\mathregular{s^{−1}} \mathregular{Pa^{−3}}$]')
    plt.legend(loc='upper right', borderaxespad=0.)

    plt.margins(0.05)
    plt.show()

# Make a file with all configurations
output_data_path = os.path.join(MAIN_PATH,
                                'input_data/')

# Get OGGM default parameters
default_glen_a = cfg.PARAMS['inversion_glen_a']
zero_fs = cfg.PARAMS['inversion_fs']
default_fs = 5.7e-20

# Make a k array
a = np.tile([k1, k2], 4)
k_array = np.append([a], [k1_low, k2_up, k2, k1, k2])

# Make a glen a array
# observations
b_0 = np.repeat(default_glen_a, 4)
b_1 = [glena_0, glena_1, glena_2, glena_3]
b = np.append(b_0, b_1)
#uncertainties
c_0 = np.repeat(default_glen_a, 2)
c_1 = [glena_1_lw, glena_2_up, default_glen_a]
c = np.append(c_0, c_1)

glen_a_array = np.append(b,c)

# Make a fs array
d0 = np.tile(np.repeat([zero_fs, default_fs], 2),2)
d1 = np.tile([zero_fs, default_fs], 2)

d3 = np.append(d0,d1)
fs_array = np.append([d3], [fs_0])

dc = pd.DataFrame({'k_calving': k_array,
                   'glen_a': glen_a_array,
                   'fs': fs_array})

dc.index = np.arange(1,len(dc)+1)
dc.index.name = 'config'

print(dc)
dc.to_csv(os.path.join(output_data_path, 'configurations'+'.csv'))
