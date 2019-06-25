# This script will plot the workflow for the columbia glacier and the LeConte
# glacier.
"""Useful plotting functions"""
import os
import geopandas as gpd
import shutil
from oggm import cfg, graphics, utils, workflow, tasks
from oggm.workflow import execute_entity_task
from matplotlib import rcParams
import matplotlib.pyplot as plt


cfg.initialize(logging_level='WORKFLOW')

WORKING_DIR = utils.gettempdir(dirname='OGGM-workflow', reset=False)

cfg.PATHS['working_dir'] = WORKING_DIR

MAIN_PATH = os.path.expanduser('~/cryo_calving_2019/')

plot_path = os.path.join(MAIN_PATH, 'plots/')

RGI_FILE = os.path.join(MAIN_PATH,
                     'input_data/01_rgi60_Alaska_modify/01_rgi60_Alaska.shp')

#This path needs to be change
Columbia_prepro = os.path.join(MAIN_PATH,
            'output_data/1_Columbia_Glacier_runs/1_preprocessing/per_glacier/')

Columbia_dir = os.path.join(Columbia_prepro,
                            'RGI60-01/RGI60-01.10/RGI60-01.10689')


# Use multiprocessing
cfg.PARAMS['use_multiprocessing'] = True
cfg.PARAMS['use_intersects'] = False
# We make the border 20 so we can use the Columbia itmix DEM
cfg.PARAMS['border'] = 20

# Set to True for operational runs
cfg.PARAMS['continue_on_error'] = True
cfg.PARAMS['inversion_fs'] = 5.7e-20
cfg.PARAMS['use_tar_shapefiles'] = False


# RGI file, 4 Marine-terminating glaciers were merged with their respective
# branches, this is why we use our own version of RGI v6
rgidf = gpd.read_file(RGI_FILE)

glaciers_to_run = ['RGI60-01.10689', 'RGI60-01.03622']
keep_indexes = [(i in glaciers_to_run) for i in rgidf.RGIId]
rgidf = rgidf.iloc[keep_indexes]

# Pre-download other files which will be needed later
_ = utils.get_cru_file(var='tmp')
p = utils.get_cru_file(var='pre')
print('CRU file: ' + p)

# Some controllers maybe this is not necessary
RUN_GIS_mask = False
RUN_GIS_PREPRO = False # run GIS pre-processing tasks (before climate)
RUN_CLIMATE_PREPRO = False # run climate pre-processing tasks
RUN_INVERSION = False  # run bed inversion

# Sort for more efficient parallel computing
rgidf = rgidf.sort_values('Area', ascending=False)


# Go - initialize working directories
# -----------------------------------
gdirs = workflow.init_glacier_regions(rgidf)

if RUN_GIS_mask:
    execute_entity_task(tasks.glacier_masks, gdirs)

#We copy Columbia glacier dir with the itmix dem
shutil.rmtree(os.path.join(WORKING_DIR,
                           'per_glacier/RGI60-01/RGI60-01.10/RGI60-01.10689'))
shutil.copytree(Columbia_dir, os.path.join(WORKING_DIR,
                            'per_glacier/RGI60-01/RGI60-01.10/RGI60-01.10689'))

# Pre-processing tasks
task_list = [
    tasks.compute_centerlines,
    tasks.initialize_flowlines,
    tasks.compute_downstream_line,
    tasks.compute_downstream_bedshape,
    tasks.catchment_area,
    tasks.catchment_intersections,
    tasks.catchment_width_geom,
    tasks.catchment_width_correction,
]

if RUN_GIS_PREPRO:
    for task in task_list:
        execute_entity_task(task, gdirs)

if RUN_CLIMATE_PREPRO:
    for gdir in gdirs:
        gdir.inversion_calving_rate = 0

    execute_entity_task(tasks.process_cru_data, gdirs)
    execute_entity_task(tasks.local_t_star, gdirs)
    execute_entity_task(tasks.mu_star_calibration, gdirs)

if RUN_INVERSION:
    # Inversion tasks
    execute_entity_task(tasks.prepare_for_inversion, gdirs, add_debug_var=True)
    execute_entity_task(tasks.mass_conservation_inversion, gdirs,
                        fs = cfg.PARAMS['inversion_fs'])



import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid, ImageGrid
from numpy.random import rand

fig = plt.figure(figsize=(14, 12))

rcParams['axes.labelsize'] = 18
rcParams['xtick.labelsize'] = 20
rcParams['ytick.labelsize'] = 20

grid1 = ImageGrid(fig, 211,
                nrows_ncols = (1, 2),
                axes_pad=1.5,
                share_all=True,
                cbar_location="left",
                cbar_mode="single",
                cbar_size="7%",
                cbar_pad=0.05,
                aspect = True
                )

llkw = {'interval': 0}
letkm = dict(color='black', ha='left', va='top', fontsize=20,
             bbox=dict(facecolor='white', edgecolor='black'))

graphics.plot_centerlines(gdirs[0], ax=grid1[0], title='', add_colorbar=True,
                          lonlat_contours_kwargs=llkw,
                          cbar_ax=grid1[0].cax, add_scalebar=True)

xt, yt = 2.45, 2.45
grid1[0].text(xt, yt, 'a', **letkm)


graphics.plot_catchment_width(gdirs[0], ax=grid1[1], title='', corrected=True,
                              add_colorbar=False,
                              lonlat_contours_kwargs=llkw,
                              add_scalebar=False)

grid1[1].text(xt, yt, 'c', **letkm)


grid1.axes_all

grid2 = ImageGrid(fig, 212,
                nrows_ncols = (1, 2),
                axes_pad=1.5,
                share_all=True,
                cbar_location="left",
                cbar_mode="each",
                cbar_size="7%",
                cbar_pad=1.5,
                aspect=True)


graphics.plot_centerlines(gdirs[1], ax=grid2[0], title='', add_colorbar=True,
                          lonlat_contours_kwargs=llkw,
                          cbar_ax=grid2[0].cax, add_scalebar=True)

grid2[0].text(xt, yt, 'b', **letkm)


graphics.plot_inversion(gdirs[1], ax=grid2[1], title='', add_colorbar=True,
                        linewidth=2, lonlat_contours_kwargs=llkw,
                        cbar_ax=grid2[1].cax,
                        add_scalebar=False)

grid2[1].text(xt, yt, 'd', **letkm)




grid2.axes_all

plt.tight_layout()
#plt.show()
plt.savefig(os.path.join(plot_path,'workflow_together.pdf'),
                         dpi=150, bbox_inches='tight')
