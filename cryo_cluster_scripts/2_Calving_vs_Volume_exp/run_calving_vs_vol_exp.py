# TODO: read this message carefully before the run!
# This script runs the Calving vs Volume experiment in Alaska with sliding
# for smoother results
# This script requires the pre-processing folder of the Columbia Glacier
# named: 1_preprocessing/per_glacier/
# This folder can be found in
# ~/home/users/your_user_name/cryo_calving_2018/output_data/1_Columbia_Glacier_runs
# This run takes approximately 2 hrs because the calving array length is 25

from __future__ import division

# Module logger
import logging
log = logging.getLogger(__name__)

# Python imports
import os
import geopandas as gpd
import numpy as np
import pandas as pd
import shutil

# Locals
import oggm.cfg as cfg
from oggm import workflow
from oggm import tasks
from oggm.workflow import execute_entity_task
from oggm import utils
from oggm.core.climate import mb_yearly_climate_on_height

# Time
import time
start = time.time()

# Regions:
# Alaska
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

#This path needs to be change
Columbia_prepro = os.path.join(MAIN_PATH,
            'output_data/1_Columbia_Glacier_runs/1_preprocessing/per_glacier/')

Columbia_dir = os.path.join(Columbia_prepro,
                            'RGI60-01/RGI60-01.10/RGI60-01.10689')

cfg.PATHS['working_dir'] = WORKING_DIR

# Use multiprocessing
cfg.PARAMS['use_multiprocessing'] = True
cfg.PARAMS['use_intersects'] = True
# We make the border 20 so we can use the Columbia itmix DEM
cfg.PARAMS['border'] = 20

# Set to True for operational runs
cfg.PARAMS['continue_on_error'] = True
cfg.PARAMS['inversion_fs'] = 5.7e-20
cfg.PARAMS['use_tar_shapefiles'] = False

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

# Some controllers maybe this is not necessary
RUN_GIS_mask = True
RUN_GIS_PREPRO = True # run GIS pre-processing tasks (before climate)
RUN_CLIMATE_PREPRO = True # run climate pre-processing tasks
RUN_INVERSION = True # run bed inversion
With_calving = False

# Run only for Lake Terminating and Marine Terminating
glac_type = [0]
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

# Compile output
#utils.compile_glacier_statistics(gdirs, filesuffix='_tw_alaska_no_calving_with_sliding_')
#utils.compile_climate_statistics(gdirs, filesuffix='_tw_alaska_')

prcp_t = []
ids = []

for gdir in gdirs:

    #Get the total precipitation of the glacier to store it later
    heights, widths = gdir.get_inversion_flowline_hw()

    df = gdir.read_json('local_mustar')
    tstar = df['t_star']
    mu_hp = int(cfg.PARAMS['mu_star_halfperiod'])
    yr = [tstar - mu_hp, tstar + mu_hp]

    years, temp, prcp = mb_yearly_climate_on_height(gdir, heights,
                                                    year_range=yr,
                                                    flatten=False)
    prcp_avg = np.average(prcp, axis=1)

    # compute the area of each section
    fls = gdir.read_pickle('inversion_flowlines')
    area_sec = widths * fls[0].dx * gdir.grid.dx

    prcpsol = np.sum(prcp_avg * area_sec)

    rho = cfg.PARAMS['ice_density']

    # We will save this!
    accu_ice = (prcpsol * 1e-9) / rho

    gdir.inversion_calving_rate = 0

    cfg.PARAMS['clip_mu_star'] = True
    cfg.PARAMS['min_mu_star'] = 0.0

    cl = gdir.read_pickle('inversion_output')[-1]
    assert cl['volume'][-1] == 0.

    volumes = []
    all_mus = []

    data_calving = data_calving = np.arange(0, 5, 0.2)

    for j, c in enumerate(data_calving):
        gdir.inversion_calving_rate = c

        # Recompute mu with calving
        tasks.local_t_star(gdir)
        execute_entity_task(tasks.mu_star_calibration, gdirs)
        tasks.prepare_for_inversion(gdir, add_debug_var=True)
        tasks.mass_conservation_inversion(gdir,
                                          fs=cfg.PARAMS['inversion_fs'])

        df = gdir.read_json('local_mustar')
        mu_star = df['mu_star_glacierwide']

        vol = []
        cl = gdir.read_pickle('inversion_output')
        for c in cl:
            vol.extend(c['volume'])
        vol = np.nansum(vol) * 1e-9

        volumes.append(vol)
        all_mus = np.append(all_mus, mu_star)

    print(volumes, data_calving)
    print(len(volumes), len(data_calving))

    d = {'calving_flux': data_calving, 'volume': volumes,
         'mu_star': all_mus}
    df = pd.DataFrame(data=d)
    df.to_csv(os.path.join(cfg.PATHS['working_dir'],
                           'sensitivity_calvsvol' + gdir.rgi_id + '.csv'))

    #Storing the precipitation per glacier in a different file
    prcp_t = np.append(prcp_t, accu_ice)
    ids = np.append(ids, gdir.rgi_id)

d_2 = {'RGI_ID': ids,
     'precp': prcp_t}
#
df_2 = pd.DataFrame(data=d_2)
df_2.to_csv(os.path.join(cfg.PATHS['working_dir'],
                               'precipitation_calving_glaciers'+'.csv'))