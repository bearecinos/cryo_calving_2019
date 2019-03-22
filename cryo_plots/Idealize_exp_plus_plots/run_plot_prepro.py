# Run example glacier and plot workflow
import os
import matplotlib.pyplot as plt
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

f = plt.figure(figsize=(14, 12))

from mpl_toolkits.axes_grid1 import ImageGrid

axs = ImageGrid(f, 111,  # as in plt.subplot(111)
                nrows_ncols=(1, 2),
                axes_pad=1.5,
                share_all=True,
                cbar_location="right",
                cbar_mode="each",
                cbar_size="7%",
                cbar_pad=0.05)

llkw = {'interval': 0}
letkm = dict(color='black', ha='left', va='top', fontsize=20,
             bbox=dict(facecolor='white', edgecolor='black'))

graphics.plot_centerlines(gdir, ax=axs[0], title='', add_colorbar=True,
                          lonlat_contours_kwargs=llkw,
                          cbar_ax=axs[0].cax, add_scalebar=False)
xt, yt = 2.45, 2.45

axs[0].text(xt, yt, 'a', **letkm)

graphics.plot_inversion(gdir, ax=axs[1], title='', linewidth=2,
                            lonlat_contours_kwargs=llkw, cbar_ax=axs[1].cax,
                            add_scalebar=True)

axs[1].text(xt, yt, 'b', **letkm)


#plt.show()
plt.tight_layout()
plt.savefig(os.path.join(plot_path,'workflow_Leconte.pdf'),
                        dpi=150, bbox_inches='tight')