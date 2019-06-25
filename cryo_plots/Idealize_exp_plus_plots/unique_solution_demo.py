# Run example glacier and plot Frontal ablation
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from oggm import cfg, utils, workflow, tasks, graphics
from oggm.core import inversion
from matplotlib import rcParams
rcParams['axes.labelsize'] = 20
rcParams['xtick.labelsize'] = 20
rcParams['ytick.labelsize'] = 20
rcParams['legend.fontsize'] = 20


MAIN_PATH = os.path.expanduser('~/cryo_calving_2019/')

WORKING_DIR = os.path.join(MAIN_PATH,
                           'output_data/7_Idealize_exp/')

plot_path = os.path.join(MAIN_PATH, 'plots/')

print(plot_path)

utils.mkdir(WORKING_DIR, reset=True)

cfg.initialize(logging_level='WORKFLOW')
cfg.PATHS['working_dir'] =  WORKING_DIR
cfg.PARAMS['border'] = 10

gdir = workflow.init_glacier_regions(['RGI60-01.03622'],
                                     from_prepro_level=3)[0]
### Idealize experiments
#Default calving flux
print(inversion.calving_flux_from_depth(gdir))

df = []
for thick in np.linspace(0, 500, 51):
    df.append(inversion.calving_flux_from_depth(gdir, thick=thick))

df = pd.DataFrame(df).set_index('thick')

ds = []
for thick in np.linspace(0, 500, 51):
    # This function simply computes the calving law
    out = inversion.calving_flux_from_depth(gdir, thick=thick)
    out['Thick (Calving law)'] = out.pop('thick')

    # Now we feed it back to OGGM
    gdir.inversion_calving_rate = out['flux']

    # The mass-balance has to adapt in order to create a flux
    tasks.local_t_star(gdir)
    tasks.mu_star_calibration(gdir)
    tasks.prepare_for_inversion(gdir)
    v_inv, _ = tasks.mass_conservation_inversion(gdir)

    # Now we get the OGGM ice thickness
    out['Thick (OGGM)'] = inversion.calving_flux_from_depth(gdir)['thick']

    # Add sliding (the fs value is outdated, but still)
    v_inv, _ = tasks.mass_conservation_inversion(gdir, fs=5.7e-20)
    out['Thick (OGGM with sliding)'] = inversion.calving_flux_from_depth(gdir)['thick']

    # Store
    ds.append(out)

ds = pd.DataFrame(ds)

# print(df)
# print(ds)
from scipy import optimize
from oggm.core.inversion import sia_thickness

cls = gdir.read_pickle('inversion_input')[-1]
slope = cls['slope_angle'][-1]
width = cls['width'][-1]

def to_minimize(wd):
    fl = inversion.calving_flux_from_depth(gdir, water_depth=wd)
    oggm = sia_thickness([slope], [width], np.array([fl['flux'] * 1e9 / cfg.SEC_IN_YEAR]))[0]
    return fl['thick'] - oggm


def to_minimize_with_sliding(wd):
    cfg.PARAMS['inversion_fs'] = 5.7e-20
    fl = inversion.calving_flux_from_depth(gdir, water_depth=wd)
    oggm = sia_thickness([slope], [width], np.array([fl['flux'] * 1e9 / cfg.SEC_IN_YEAR]),
                         fs=5.7e-20)[0]
    return fl['thick'] - oggm


wd = np.linspace(0.1, 500, 51)
out = []
for w in wd:
    out.append(to_minimize(w))

out_fs = []
for w in wd:
    out_fs.append(to_minimize_with_sliding(w))

dg = pd.DataFrame(list(zip(wd, out, out_fs)),
                  columns=['Water depth [m]',
                           'OGGM - Calving law',
                           'OGGM (with sliding) - Calving law'])

print(len(ds))

print(dg)


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
         label='$q_{calving}$')
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
plt.plot(ds['flux'], ds['Thick (Calving law)'],
         color=sns.xkcd_rgb["ocean blue"],
         linewidth=2.5)
plt.plot(ds['flux'], ds['Thick (OGGM)'],
         sns.xkcd_rgb["burnt orange"],
         linewidth=2.5)
plt.plot(ds['flux'], ds['Thick (OGGM with sliding)'],
         sns.xkcd_rgb["teal green"],
         linewidth=2.5)
plt.xlabel('Frontal ablation [$km³.yr^{-1}$]')
plt.ylabel('Ice thickness [m]')

plt.legend(loc='lower right', borderaxespad=0.)
at = AnchoredText('b', prop=dict(size=20), frameon=True, loc=2)
f2_ax2.add_artist(at)

f2_ax3 = fig2.add_subplot(gs[2])
plt.plot(dg['Water depth [m]'], dg['OGGM - Calving law'],
         sns.xkcd_rgb["burnt orange"],
         linewidth=2.5)
plt.plot(dg['Water depth [m]'], dg['OGGM (with sliding) - Calving law'],
         sns.xkcd_rgb["teal green"],
         linewidth=2.5)

plt.hlines([0], 0.100, 500)
plt.tick_params(axis='both', bottom=True, left=True, width=2,
                direction='out', length=5)
plt.xlabel('Water depth [m]')
plt.ylabel('Thickness difference [m]')
plt.legend(loc='upper right', borderaxespad=0., fontsize=18)

at = AnchoredText('c', prop=dict(size=20), frameon=True, loc=2)
f2_ax3.add_artist(at)


plt.subplots_adjust(hspace=0.2)
plt.margins(0.05)
plt.tight_layout()
#plt.show()
plt.savefig(os.path.join(plot_path, 'thickness_unique_solution.pdf'), dpi=150,
            bbox_inches='tight')

