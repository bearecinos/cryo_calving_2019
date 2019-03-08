import pandas as pd
import os
import geopandas as gpd
import numpy as np
from oggm import cfg, utils
from oggm import workflow
import warnings

cfg.initialize()
# Reading glacier directories per experiment

MAIN_PATH = os.path.expanduser('~/cryo_calving_2019/')

exp_dir_path = os.path.join(MAIN_PATH,
'output_data/4_Runs_different_configurations/4_3_With_calving_exp_onlyMT_vbsl')

dir_name = os.listdir(exp_dir_path)
full_dir_name = []

for d in dir_name:
    full_dir_name.append(os.path.join(exp_dir_path,d))

full_dir_name = sorted(full_dir_name)

# Reading RGI
RGI_FILE = os.path.join(MAIN_PATH,
                'input_data/01_rgi60_Alaska_modify/01_rgi60_Alaska.shp')

print(full_dir_name)

data = []

for glac_dir in full_dir_name:
    cfg.PATHS['working_dir'] = glac_dir
    cfg.PARAMS['border'] = 20

    #Read RGI file
    rgidf = gpd.read_file(RGI_FILE)

    # Run only for Marine terminating
    glac_type = [0, 2]
    keep_glactype = [(i not in glac_type) for i in rgidf.TermType]
    rgidf = rgidf.iloc[keep_glactype]

    gdirs = workflow.init_glacier_regions(rgidf)

    vbsl_no_calving_per_dir = []
    vbsl_calving_per_dir = []
    ids = []

    for gdir in gdirs:

        vbsl_no_calving_per_glacier = []
        vbsl_calving_per_glacier = []

        #Get the data that we need from each glacier
        map_dx = gdir.grid.dx

        #Get flowlines
        fls = gdir.read_pickle('inversion_flowlines')

        #Get inversion output
        inv = gdir.read_pickle('inversion_output', filesuffix='_without_calving_')
        inv_c = gdir.read_pickle('inversion_output')

        import matplotlib.pylab as plt
        for f, cl, cc, in zip(range(len(fls)), inv , inv_c):

            x = np.arange(fls[f].nx) * fls[f].dx * map_dx * 1e-3
            surface = fls[f].surface_h

            # Getting the thickness per branch
            thick = cl['thick']
            vol = cl['volume']

            thick_c = cc['thick']
            vol_c = cc['volume']

            ## TODO: delete this part because now I dont have the flowline shortage
            ## problem
            # if len(surface) != len(thick):
            #     plt.figure()
            #     plt.plot(surface, color='r')
            #     plt.plot(thick, color='grey')
            #     plt.title(
            #         'Glacier ' + gdir.rgi_id + ' flowline No.' + np.str(f) +
            #         ' out of ' + np.str(len(fls)))
            #     plt.savefig(os.path.join(cfg.PATHS['working_dir'],
            #                              gdir.rgi_id + '_' + np.str(
            #                                  f) + '.png'))
            #     # Warning('({}) something went wrong with the '
            #     #               'inversion'.format(gdir.rgi_id))
            #     pass
            # else:
            bed = surface - thick
            bed_c = surface - thick_c

            # Find volume below sea level without calving in km³
            index_sl = np.where(bed < 0.0)
            vol_sl = sum(vol[index_sl]) / 1e9
            #print('before calving',vol_sl)

            index_sl_c = np.where(bed_c < 0.0)
            vol_sl_c = sum(vol_c[index_sl_c]) / 1e9
            #print('after calving',vol_sl_c)

            vbsl_no_calving_per_glacier = np.append(
             vbsl_no_calving_per_glacier, vol_sl)

            vbsl_calving_per_glacier = np.append(
             vbsl_calving_per_glacier, vol_sl_c)

            ids = np.append(ids, gdir.rgi_id)

        # We sum up all the volume below sea level in all branches
        vbsl_no_calving_per_glacier = sum(vbsl_no_calving_per_glacier)
        vbsl_calving_per_glacier = sum(vbsl_calving_per_glacier)

        vbsl_no_calving_per_dir = np.append(vbsl_no_calving_per_dir,
                                        vbsl_no_calving_per_glacier)

        vbsl_calving_per_dir = np.append(vbsl_calving_per_dir,
                                     vbsl_calving_per_glacier)

        np.set_printoptions(suppress=True)


    d = {'RGIId': pd.unique(ids),
     'volume bsl': vbsl_no_calving_per_dir,
     'volume bsl with calving': vbsl_calving_per_dir}
    data_frame = pd.DataFrame(data=d)
    data_frame.to_csv(os.path.join(glac_dir,'volume_below_sea_level.csv'))