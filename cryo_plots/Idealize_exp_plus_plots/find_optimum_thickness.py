# Run example glacier and plot workflow
import os
import matplotlib.pyplot as plt
from oggm import cfg, utils, workflow, tasks, graphics
from matplotlib import rcParams
rcParams['axes.labelsize'] = 20
rcParams['xtick.labelsize'] = 20
rcParams['ytick.labelsize'] = 20
rcParams['legend.fontsize'] = 15
from matplotlib.offsetbox import AnchoredText
import seaborn as sns


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

#### Finding the optimum thickness
dr = utils.find_inversion_calving(gdir)

gdir.inversion_calving_rate = 0
cfg.PARAMS['k_calving'] = 10  # default is 2.4
dm = utils.find_inversion_calving(gdir)


f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2,
                                           figsize=(14, 10))

sns.set_style("white")
from matplotlib.ticker import MaxNLocator

ax1.plot(dr['calving_flux'], linewidth=2.5, label='$F_{calving}$, $k$ = 2.4 yr$^{-1}$')
ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
ax1.tick_params(axis='both', bottom=True, left=True, width=2, direction='out', length=5)

ax1.set_ylabel('Frontal ablation [$km³.yr^{-1}$]')
ax1.set_xlabel('Iteration step')
at = AnchoredText('a', prop=dict(size=20), frameon=True, loc=2)
ax1.add_artist(at)
ax1.legend(loc='lower right', borderaxespad=0.)
plt.margins(0.05)


ax2.plot(dr['mu_star'], linewidth=2.5, color='C1', label='$\mu^*$')
ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
ax2.tick_params(axis='both', bottom=True, left=True, width=2, direction='out', length=5)
ax2.set_ylabel('Temperature sensitivity \n [mm yr$^{-1}$ K$^{-1}$]')
ax2.set_xlabel('Iteration step')
at = AnchoredText('b', prop=dict(size=20), frameon=True, loc=2)
ax2.add_artist(at)
ax2.legend(loc='upper right', borderaxespad=0.)
plt.margins(0.05)

ax3.plot(dm['calving_flux'], linewidth=2.5, label='$F_{calving}$, $k$ = 10.0 yr$^{-1}$')
ax3.tick_params(axis='both', bottom=True, left=True, width=2, direction='out', length=5)
ax3.set_ylabel('Frontal ablation [$km³.yr^{-1}$]')
ax3.set_xlabel('Iteration step')
at = AnchoredText('c', prop=dict(size=20), frameon=True, loc=2)
ax3.add_artist(at)
ax3.legend(loc='lower right', borderaxespad=0.)
plt.margins(0.05)

ax4.plot(dm['mu_star'], linewidth=2.5, color='C1', label='$\mu^*$')
ax4.tick_params(axis='both', bottom=True, left=True, width=2, direction='out', length=5)
ax4.set_ylabel('Temperature sensitivity \n [mm yr$^{-1}$ K$^{-1}$]')
ax4.set_xlabel('Iteration step')
at = AnchoredText('d', prop=dict(size=20), frameon=True, loc=2)
ax4.add_artist(at)
ax4.legend(loc='upper right', borderaxespad=0.)
plt.margins(0.05)
#
#plt.show()
plt.tight_layout()
plt.savefig(os.path.join(plot_path, 'loop.pdf'), dpi=150,
                       bbox_inches='tight')