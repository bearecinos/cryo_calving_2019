# Run example glacier and plot workflow
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from oggm import cfg, utils, workflow
from matplotlib import rcParams
rcParams['axes.labelsize'] = 20
rcParams['xtick.labelsize'] = 20
rcParams['ytick.labelsize'] = 20
rcParams['legend.fontsize'] = 20

width_cm = 12
height_cm = 6


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


## Different water depths exp
dg = pd.DataFrame()
dg['$d$ = 001'] = utils.find_inversion_calving(gdir, water_depth=1)['calving_flux']
dg['$d$ = 100'] = utils.find_inversion_calving(gdir, water_depth=100)['calving_flux']
dg['$d$ = 200'] = utils.find_inversion_calving(gdir, water_depth=200)['calving_flux']
dg['$d$ = 300'] = utils.find_inversion_calving(gdir, water_depth=300)['calving_flux']
dg['$d$ = 400'] = utils.find_inversion_calving(gdir, water_depth=400)['calving_flux']
dg['$d$ = 500'] = utils.find_inversion_calving(gdir, water_depth=500)['calving_flux']

#f = plt.figure(figsize=(width_cm, height_cm))


sns.set_color_codes("colorblind")
sns.set_style("white")
dg.iloc[1:].plot(figsize=(width_cm, height_cm))

plt.tick_params(axis='both', bottom=True, left=True, width=2, direction='out', length=5)
plt.xlabel('Iterations')
plt.ylabel('Frontal ablation [$km³.yr^{-1}$]')
plt.legend(loc='upper right', borderaxespad=0.)
plt.margins(0.05)
plt.tight_layout()
plt.savefig(os.path.join(plot_path, 'different_wd.pdf'), dpi=150,
                      bbox_inches='tight')

