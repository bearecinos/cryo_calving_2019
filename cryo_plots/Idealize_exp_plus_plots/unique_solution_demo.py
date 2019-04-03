# Run example glacier and plot workflow
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from oggm import cfg, utils, workflow, tasks, graphics
from matplotlib import rcParams
rcParams['axes.labelsize'] = 20
rcParams['xtick.labelsize'] = 20
rcParams['ytick.labelsize'] = 20
rcParams['legend.fontsize'] = 20


MAIN_PATH = os.path.expanduser('~/cryo_calving_2019/')

WORKING_DIR = os.path.join(MAIN_PATH,
                           'output_data/7_Idealize_exp/')

plot_path = os.path.join(MAIN_PATH, 'plots/')

utils.mkdir(WORKING_DIR, reset=True)

cfg.initialize(logging_level='WORKFLOW')
cfg.PATHS['working_dir'] =  WORKING_DIR
cfg.PARAMS['border'] = 10

gdir = workflow.init_glacier_regions(['RGI60-01.03622'],
                                     from_prepro_level=3)[0]
### Idealize experiments
#default calving flux
print(utils.calving_flux_from_depth(gdir))

df = []
for thick in np.linspace(0, 500, 51):
    df.append(utils.calving_flux_from_depth(gdir, thick=thick))

df = pd.DataFrame(df).set_index('thick')

ds = []
for thick in np.linspace(0, 500, 51):
    # This function simply computes the calving law
    out = utils.calving_flux_from_depth(gdir, thick=thick)
    out['Thick (prescribed)'] = out.pop('thick')

    # Now we feed it back to OGGM
    gdir.inversion_calving_rate = out['flux']

    # The mass-balance has to adapt in order to create a flux
    tasks.local_t_star(gdir)
    tasks.mu_star_calibration(gdir)
    tasks.prepare_for_inversion(gdir)
    v_inv, _ = tasks.mass_conservation_inversion(gdir)

    # Now we get the OGGM ice thickness
    out['Thick (OGGM)'] = utils.calving_flux_from_depth(gdir)['thick']

    # Add sliding (the fs value is outdated, but still)
    v_inv, _ = tasks.mass_conservation_inversion(gdir, fs=5.7e-20)
    out['Thick (OGGM with sliding)'] = utils.calving_flux_from_depth(gdir)['thick']

    # Store
    ds.append(out)

ds = pd.DataFrame(ds)

#print(df)
#print(ds)


## Different water depths exp
dg = pd.DataFrame()
dg['$d$ = 001 m'] = utils.find_inversion_calving(gdir, water_depth=1)['calving_flux']
dg['$d$ = 100 m'] = utils.find_inversion_calving(gdir, water_depth=100)['calving_flux']
dg['$d$ = 200 m'] = utils.find_inversion_calving(gdir, water_depth=200)['calving_flux']
dg['$d$ = 300 m'] = utils.find_inversion_calving(gdir, water_depth=300)['calving_flux']
dg['$d$ = 400 m'] = utils.find_inversion_calving(gdir, water_depth=400)['calving_flux']
dg['$d$ = 500 m'] = utils.find_inversion_calving(gdir, water_depth=500)['calving_flux']

from matplotlib import gridspec
from matplotlib.offsetbox import AnchoredText
from matplotlib.ticker import MaxNLocator

width_cm = 14
height_cm = 7

# Create figure and axes instances
fig2 = plt.figure(2, figsize=(width_cm, height_cm*3))
gs = gridspec.GridSpec(3, 1, hspace=0.2)

#sns.set_color_codes("colorblind")
f2_ax1 = fig2.add_subplot(gs[0])
sns.set_style("white")

f2_ax1.tick_params(axis='both', bottom=True, left=True, width=2,
                   direction='out', length=5)

plt.plot(df['flux'], color=sns.xkcd_rgb["ocean blue"], linewidth=2.5,
         label='$F_{calving}$')
plt.ylabel('Frontal ablation [$km³.yr^{-1}$]')
plt.xlabel('Ice thickness [m]')
plt.legend(loc='lower right', borderaxespad=0.)

at = AnchoredText('a', prop=dict(size=20), frameon=True, loc=2)
f2_ax1.add_artist(at)

f2_ax2 = fig2.add_subplot(gs[1])
sns.set_color_codes("colorblind")
sns.set_style("white")

f2_ax2.tick_params(axis='both', bottom=True, left=True, width=2,
                   direction='out', length=5)
plt.plot(ds['flux'], ds['Thick (prescribed)'], linewidth=2.5)
plt.plot(ds['flux'], ds['Thick (OGGM)'], linewidth=2.5)
plt.plot(ds['flux'], ds['Thick (OGGM with sliding)'], linewidth=2.5)

plt.xlabel('Frontal ablation [$km³.yr^{-1}$]')
plt.ylabel('Ice thickness [m]')

plt.legend(loc='lower right', borderaxespad=0.)
at = AnchoredText('b', prop=dict(size=20), frameon=True, loc=2)
f2_ax2.add_artist(at)

f2_ax3 = fig2.add_subplot(gs[2])
dg.iloc[1:].plot(ax=f2_ax3,  linewidth=2.5)
plt.tick_params(axis='both', bottom=True, left=True, width=2,
                direction='out', length=5)
plt.xlabel('Iteration step')
plt.ylabel('Frontal ablation [$km³.yr^{-1}$]')
plt.legend(loc='upper right', borderaxespad=0., fontsize=18)

at = AnchoredText('c', prop=dict(size=20), frameon=True, loc=2)
f2_ax3.add_artist(at)


plt.subplots_adjust(hspace=0.2)
plt.margins(0.05)
plt.tight_layout()
#plt.show()
plt.savefig(os.path.join(plot_path, 'thickness_unique_solution.pdf'), dpi=150,
            bbox_inches='tight')

