# This will run OGGM on the Alaska Region without calving
from __future__ import division

# Module logger
import logging
log = logging.getLogger(__name__)

# Python imports
import os
import geopandas as gpd
import shutil

# Locals
import oggm.cfg as cfg
from oggm import workflow
from oggm import tasks
from oggm.workflow import execute_entity_task
from oggm import utils
from oggm.core import inversion

# Time
import time
start = time.time()

# Regions: Alaska
rgi_region = '01'

# Initialize OGGM and set up the run parameters
# ---------------------------------------------

cfg.initialize()
rgi_version = '6'

SLURM_WORKDIR = os.environ["WORKDIR"]

# Local paths (where to write output and where to download input)
WORKING_DIR = SLURM_WORKDIR

MAIN_PATH = os.path.expanduser('~/cryo_calving_2019/')

RGI_FILE = os.path.join(MAIN_PATH,
                        'input_data/01_rgi60_Alaska_modify/01_rgi60_Alaska.shp')

Columbia_prepro = os.path.join(MAIN_PATH,
          'output_data/1_Columbia_Glacier_runs/1_preprocessing/per_glacier/')

Columbia_dir = os.path.join(Columbia_prepro,
                            'RGI60-01/RGI60-01.10/RGI60-01.10689')


cfg.PATHS['working_dir'] = WORKING_DIR
# Use multiprocessing
cfg.PARAMS['use_multiprocessing'] = True
cfg.PARAMS['use_intersects'] = True

# We make the border 20
cfg.PARAMS['border'] = 20
# Set to False for operating the calving loop
cfg.PARAMS['continue_on_error'] = False
cfg.PARAMS['use_tar_shapefiles'] = False

# Inversion and calving configuration
cfg.PARAMS['inversion_fs'] = 5.7e-20
cfg.PARAMS['k_calving'] = 0.6659
# Default glen a

# We use intersects
path = utils.get_rgi_intersects_region_file(rgi_region, version=rgi_version)
cfg.set_intersects_db(path)

# RGI file, 4 Marine-terminating glaciers were merged with their respective
# branches, this is why we use our own version of RGI v6
rgidf = gpd.read_file(RGI_FILE)

# Pre-download other files which will be needed later
_ = utils.get_cru_file(var='tmp')
p = utils.get_cru_file(var='pre')
print('CRU file: ' + p)

# Some globals for more control on what to run
RUN_GIS_mask = True
RUN_GIS_PREPRO = True # run GIS pre-processing tasks (before climate)
RUN_CLIMATE_PREPRO = True # run climate pre-processing tasks
RUN_INVERSION = True  # run bed inversion

# Run only for Marine terminating
glac_type = [0, 2]
keep_glactype = [(i not in glac_type) for i in rgidf.TermType]
rgidf = rgidf.iloc[keep_glactype]

# Sort for more efficient parallel computing
rgidf = rgidf.sort_values('Area', ascending=False)

log.info('Starting run for RGI reg: ' + rgi_region)
log.info('Number of glaciers: {}'.format(len(rgidf)))

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
    execute_entity_task(tasks.prepare_for_inversion, gdirs,
                        add_debug_var=True)
    execute_entity_task(tasks.mass_conservation_inversion, gdirs)


# Compile output
utils.compile_glacier_statistics(gdirs, filesuffix='config_4_')

# Log
m, s = divmod(time.time() - start, 60)
h, m = divmod(m, 60)
log.info("OGGM no_calving is done! Time needed: %02d:%02d:%02d" % (h, m, s))