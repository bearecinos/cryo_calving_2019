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
obs = np.repeat(15.11 * 1.091, len(k))

#Find points below and above observations
# k1
print('FOR K1______')
print('lower', data_frame1[data_frame1['calving_flux'] < obs[0]].index.values)
print('equal', data_frame1[data_frame1['calving_flux'] == obs[0]].index.values)
print('bigger', data_frame1[data_frame1['calving_flux'] > obs[0]].index.values)

# k2
print('FOR K2______')
print('lower', data_frame2[data_frame2['calving_flux'] < obs[0]].index.values)
print('equal', data_frame2[data_frame2['calving_flux'] == obs[0]].index.values)
print('bigger', data_frame2[data_frame2['calving_flux'] > obs[0]].index.values)


#Defining the points
points_k1 = np.asarray([k[8], data_frame1.iloc[8]['calving_flux'],
          k[9], data_frame1.iloc[9]['calving_flux']])

points_k2 = np.asarray([k[8], data_frame2.iloc[8]['calving_flux'],
          k[9], data_frame2.iloc[9]['calving_flux']])

points_obs = np.asanyarray([k[0], obs[0],
          k[1], obs[1]])

print('Points k1',points_k1)
print('Points k2',points_k2)
print('Points obs', points_obs)

# Finding slope and intercept for the lines of each data frame
m1 = getSlope(points_k1[0], points_k1[1], points_k1[2], points_k1[3])
b1 = getYInt(points_k1[0], points_k1[1], points_k1[2], points_k1[3])

m2 = getSlope(points_k2[0], points_k2[1], points_k2[2], points_k2[3])
b2 = getYInt(points_k2[0], points_k2[1], points_k2[2], points_k2[3])

m3 = getSlope(points_obs[0], points_obs[1], points_obs[2], points_obs[3])
b3 = getYInt(points_obs[0], points_obs[1], points_obs[2], points_obs[3])

print('______Equation for k1_____')
print('slope',m1)
print('Y intercept',b1)
print('______Equation for k2_____')
print('slope',m2)
print('Y intercept',b2)
print('______Equation for obs_____')
print('slope',m3)
print('Y intercept',b3)

k1 = np.around(get_lines_Int(m1,b1,m3,b3),5)
k2 = np.around(get_lines_Int(m2,b2,m3,b3),5)
print('k1', k1)
print('k2', k2)


#########  Plotting things
rcParams['axes.labelsize'] = 20
rcParams['xtick.labelsize'] = 20
rcParams['ytick.labelsize'] = 20
rcParams['legend.fontsize'] = 18
# Set figure width and height in cm
width_cm = 14
height_cm = 7

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

plt.plot(k, b3 + m3*k, '--', color='black', linewidth=3.0,
        label='Frontal ablation (McNabb et al., 2015)')

plt.plot(k1, obs[0], 'x', markersize=20,
         color=sns.xkcd_rgb["ocean blue"], linewidth=4,
         label='k1 ='+ str(round(k1, 2)))

plt.plot(k2, obs[0], 'x', markersize=20,
         color=sns.xkcd_rgb["teal green"], linewidth=4,
         label='k2 =' + str(round(k2, 2)))


plt.fill_between(k, (b3 + m3*k) - 3.96, (b3 + m3*k) + 3.96,
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

print(filenames_A)

# Glen_a array or FS
factors = np.arange(0.6,2.0,0.02)
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

#### Finding intercepts with observations #######################

#Find points below and above observations
# glen_a1
print('FOR glen_A_0_______')
print('lower', data_frame0_A[data_frame0_A['calving_flux'] < obs[0]].index.values)
print('equal', data_frame0_A[data_frame0_A['calving_flux'] == obs[0]].index.values)
print('bigger', data_frame0_A[data_frame0_A['calving_flux'] > obs[0]].index.values)

# glen_a2
print('FOR glen_A_1______')
print('lower', data_frame1_A[data_frame1_A['calving_flux'] < obs[0]].index.values)
print('equal', data_frame1_A[data_frame1_A['calving_flux'] == obs[0]].index.values)
print('bigger', data_frame1_A[data_frame1_A['calving_flux'] > obs[0]].index.values)

#glen_a3
print('FOR glen_A_2______')
print('lower', data_frame2_A[data_frame2_A['calving_flux'] < obs[0]].index.values)
print('equal', data_frame2_A[data_frame2_A['calving_flux'] == obs[0]].index.values)
print('bigger', data_frame2_A[data_frame2_A['calving_flux'] > obs[0]].index.values)

#glen_a4
print('FOR glen_A_3______')
print('lower', data_frame3_A[data_frame3_A['calving_flux'] < obs[0]].index.values)
print('equal', data_frame3_A[data_frame3_A['calving_flux'] == obs[0]].index.values)
print('bigger', data_frame3_A[data_frame3_A['calving_flux'] > obs[0]].index.values)


#Defining the points
points_glena_0 = np.asarray([glen_a[20], data_frame0_A.iloc[20]['calving_flux'],
          glen_a[21], data_frame0_A.iloc[21]['calving_flux']])

points_glena_1 = np.asarray([glen_a[26], data_frame1_A.iloc[26]['calving_flux'],
          glen_a[27], data_frame1_A.iloc[27]['calving_flux']])

points_glena_2 = np.asarray([glen_a[14], data_frame2_A.iloc[14]['calving_flux'],
          glen_a[15], data_frame2_A.iloc[15]['calving_flux']])

points_glena_3 = np.asarray([glen_a[20], data_frame3_A.iloc[20]['calving_flux'],
          glen_a[21], data_frame3_A.iloc[21]['calving_flux']])

points_glena_obs = np.asanyarray([glen_a[0], obs[0],
          glen_a[1], obs[1]])

# See the points between which we will interpolate
print('Points glen_0', points_glena_0)
print('Points glen_1', points_glena_1)
print('Points glen_2', points_glena_2)
print('Points glen_3', points_glena_3)
print('Points obs', points_obs)

# Finding slope and intercept for the lines of each data frame
m0_a = getSlope(points_glena_0[0], points_glena_0[1],
                points_glena_0[2], points_glena_0[3])
b0_a = getYInt(points_glena_0[0], points_glena_0[1],
               points_glena_0[2], points_glena_0[3])

m1_a = getSlope(points_glena_1[0], points_glena_1[1],
                points_glena_1[2], points_glena_1[3])
b1_a = getYInt(points_glena_1[0], points_glena_1[1],
               points_glena_1[2], points_glena_1[3])

m2_a = getSlope(points_glena_2[0], points_glena_2[1],
                points_glena_2[2], points_glena_2[3])
b2_a = getYInt(points_glena_2[0], points_glena_2[1],
               points_glena_2[2], points_glena_2[3])

m3_a = getSlope(points_glena_3[0], points_glena_3[1],
                points_glena_3[2], points_glena_3[3])
b3_a = getYInt(points_glena_3[0], points_glena_3[1],
               points_glena_3[2], points_glena_3[3])

m4_a = getSlope(points_obs[0], points_obs[1],
                points_obs[2], points_obs[3])
b4_a = getYInt(points_obs[0], points_obs[1],
               points_obs[2], points_obs[3])

print('______Equation for glen 0_____')
print('slope', m0_a)
print('Y intercept',b0_a)
print('______Equation for glen 1_____')
print('slope',m1_a)
print('Y intercept',b1_a)
print('______Equation for glen 2 _____')
print('slope',m2_a)
print('Y intercept',b2_a)
print('______Equation for glen 3 _____')
print('slope',m3_a)
print('Y intercept',b3_a)
print('______Equation for obs _____')
print('slope',m4_a)
print('Y intercept', b4_a)

# Get intercepts
glena_0 = get_lines_Int(m0_a, b0_a, m4_a, b4_a)
glena_1 = get_lines_Int(m1_a, b1_a, m4_a, b4_a)
glena_2 = get_lines_Int(m2_a, b2_a, m4_a, b4_a)
glena_3 = get_lines_Int(m3_a, b3_a, m4_a, b4_a)

print('glen_0', glena_0)
print('glen_1', glena_1)
print('glen_2', glena_2)
print('glen_3', glena_3)

###########################plot glen a experiments ################################

fig = plt.figure(2, figsize=(width_cm, height_cm))

my_labels_glena = {"x0": "fs = 0.0, " + 'k1 = '+ str(round(k1, 2)),
                   "x1": "fs = 0.0, " + 'k2 = '+ str(round(k2, 2)),
                   "x2": "fs = OGGM default, " + 'k1 = ' + str(round(k1, 2)),
                   "x3": "fs = OGGM default, " + 'k2 = ' + str(round(k2, 2))}
sns.set_color_codes("colorblind")
sns.set_style("white")


plt.plot(glen_a, data_frame0_A, linestyle="--", #color=sns.xkcd_rgb["forest green"],
             label=my_labels_glena["x0"], linewidth=2.5)

plt.plot(glen_a, data_frame1_A, linestyle="--", #color=sns.xkcd_rgb["green"],
             label=my_labels_glena["x1"], linewidth=2.5)

plt.plot(glen_a, data_frame2_A, #color=sns.xkcd_rgb["teal"],
             label=my_labels_glena["x2"], linewidth=2.5)

plt.plot(glen_a, data_frame3_A, #color=sns.xkcd_rgb["turquoise"],
             label=my_labels_glena["x3"], linewidth=2.5)


plt.plot(glena_0, obs[0], 'x', markersize=20, linewidth=4)

plt.plot(glena_1, obs[0], 'x', markersize=20,
         linewidth=4)

plt.plot(glena_2, obs[0], 'x', markersize=20,
         linewidth=4)

plt.plot(glena_3, obs[0], 'x', markersize=20,
         linewidth=4)

plt.plot(glen_a, np.repeat(15.11*1.091, len(glen_a)), '--', color='black',
             label='Frontal ablation (McNabb et al., 2015)', linewidth=3.0)

plt.fill_between(glen_a,np.repeat(15.11*1.091-3.96, len(glen_a)),
                 np.repeat(15.11*1.091+3.96, len(glen_a)),
                 color=sns.xkcd_rgb["grey"], alpha=0.3)

#plt.xticks(glen_a)
#plt.gca().axes.get_xticklines()
#plt.yticks(np.arange(0, 35, 3.0))


plt.ylabel('Alaska frontal ablation \n [$km³.yr^{-1}$]')
plt.xlabel('Glen A [$\mathregular{s^{−1}} \mathregular{Pa^{−3}}$]')
plt.legend(loc='upper right', borderaxespad=0.)
letkm = dict(color='black', ha='left', va='top', fontsize=20,
                 bbox=dict(facecolor='white', edgecolor='black'))
#plt.text(glen_a[0]-1.15e-25, 27.25, 'b', **letkm)

plt.margins(0.05)
plt.show()
