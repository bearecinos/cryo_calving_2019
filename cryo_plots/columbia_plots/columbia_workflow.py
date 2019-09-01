# This script will plot the workflow for the columbia glacier, the calving output
# and the columbia bed profile along the main centreline

"""Useful plotting functions"""
import netCDF4
import pandas as pd
import shapely.geometry as shpg
from matplotlib import rcParams
from scipy import stats
import os
import geopandas as gpd
import numpy as np
import salem

from oggm import cfg, graphics

import oggm.utils
import matplotlib.pyplot as plt


cfg.initialize()


MAIN_PATH = os.path.expanduser('~/cryo_calving_2019/')

WORKING_DIR = os.path.join(MAIN_PATH,
                           'output_data/1_Columbia_Glacier_runs/2_calving/')

RGI_FILE = os.path.join(WORKING_DIR,
                'per_glacier/RGI60-01/RGI60-01.10/RGI60-01.10689/outlines.shp')

cfg.PATHS['dem_file'] = os.path.join(WORKING_DIR,
                    'per_glacier/RGI60-01/RGI60-01.10/RGI60-01.10689/dem.tif')

entity = gpd.read_file(RGI_FILE).iloc[0]

gdir = oggm.GlacierDirectory(entity,
                             base_dir=os.path.join(WORKING_DIR, 'per_glacier'))

plot_path = os.path.join(MAIN_PATH, 'plots/')

k = 2.4

# Reading and finding out the total volume
vol_total=[]
vol_total_c=[]

inv = gdir.read_pickle('inversion_output', filesuffix='_without_calving_')
inv_c = gdir.read_pickle('inversion_output')

for l, c in zip(inv, inv_c):
    vol_total.extend(l['volume'])
    vol_total_c.extend(c['volume'])

total_vol = np.nansum(vol_total) * 1e-9
print(total_vol)

total_vol_c = np.nansum(vol_total_c) * 1e-9
print(total_vol_c)

percentage = total_vol_c * 100 / total_vol

print('Volume percentage Columbia', percentage-100)

# Plotting functions
# # ----------------
@graphics._plot_map
def plot_inversion_with_calving(gdirs, ax=None, smap=None, linewidth=3,
                                vmax=None, k=None):
    """Plots the result of the inversion out of a glacier directory."""

    gdir = gdirs[0]
    with netCDF4.Dataset(gdir.get_filepath('gridded_data')) as nc:
        topo = nc.variables['topo'][:]

    # Dirty optim
    try:
        smap.set_topography(topo)
    except ValueError:
        pass

    toplot_th = np.array([])
    toplot_lines = []
    toplot_crs = []
    vol = []
    for gdir in gdirs:
        crs = gdir.grid.center_grid
        geom = gdir.read_pickle('geometries')
        inv = gdir.read_pickle('inversion_output',
                               filesuffix='_without_calving_')
        # Plot boundaries
        poly_pix = geom['polygon_pix']
        smap.set_geometry(poly_pix, crs=crs, fc='none', zorder=2,
                          linewidth=.2)
        for l in poly_pix.interiors:
            smap.set_geometry(l, crs=crs, color='black', linewidth=0.5)

        # plot Centerlines
        cls = gdir.read_pickle('inversion_flowlines')
        for l, c in zip(cls, inv):

            smap.set_geometry(l.line, crs=crs, color='gray',
                              linewidth=1.2, zorder=50)
            toplot_th = np.append(toplot_th, c['thick'])
            for wi, cur, (n1, n2) in zip(l.widths, l.line.coords, l.normals):
                l = shpg.LineString([shpg.Point(cur + wi / 2. * n1),
                                     shpg.Point(cur + wi / 2. * n2)])
                toplot_lines.append(l)
                toplot_crs.append(crs)
            vol.extend(c['volume'])

    cm = plt.cm.get_cmap('YlOrRd')
    dl = salem.DataLevels(cmap=cm, nlevels=256, data=toplot_th,
                          vmin=0, vmax=vmax)
    colors = dl.to_rgb()
    for l, c, crs in zip(toplot_lines, colors, toplot_crs):
        smap.set_geometry(l, crs=crs, color=c,
                          linewidth=linewidth, zorder=50)

    smap.plot(ax)
    return dict(cbar_label='Section thickness [m]',
                cbar_primitive=dl,
                title_comment=' ({:.2f} km3)'.format(np.nansum(vol) * 1e-9))

def mean_d(data, ref):
        "Mean difference between two arrays"
        diff = data - ref
        abs_diff = np.abs(diff)
        return np.sum(abs_diff) / (data.size * ref.size)

# figure 2  ---------------
Plot_fig_2 = False

if Plot_fig_2:

    rcParams['axes.labelsize'] = 20
    rcParams['xtick.labelsize'] = 20
    rcParams['ytick.labelsize'] = 20

    f = plt.figure(figsize=(14, 12))
    from mpl_toolkits.axes_grid1 import ImageGrid

    axs = ImageGrid(f, 111,  # as in plt.subplot(111)
                    nrows_ncols=(1, 2),
                    axes_pad=0.15,
                    share_all=True,
                    cbar_location="right",
                    cbar_mode="edge",
                    cbar_size="7%",
                    cbar_pad=0.15,
                    )

    llkw = {'interval': 0}
    letkm = dict(color='black', ha='left', va='top', fontsize=20,
                 bbox=dict(facecolor='white', edgecolor='black'))

    plot_inversion_with_calving(gdir, ax=axs[0], title='', linewidth=2,
                            add_colorbar=False, vmax=1600,
                            lonlat_contours_kwargs=llkw,
                            add_scalebar=False)
    xt, yt = 2.45, 2.45
    xvol, yvol = 185, 2.45
    axs[0].text(xt, yt, 'a', **letkm)
    axs[0].text(xvol, yvol, '{:.2f} km$^3$'.format(total_vol), **letkm)

    graphics.plot_inversion(gdir, ax=axs[1], title='', linewidth=2,
                            add_colorbar=True, vmax=1600,
                            lonlat_contours_kwargs=llkw, cbar_ax=axs[1].cax,
                            add_scalebar=True)
    axs[1].text(xt, yt, 'b', **letkm)
    axs[1].text(xvol, yvol, '{:.2f} km$^3$'.format(total_vol_c), **letkm)

    plt.tight_layout()
    #plt.show()
    plt.savefig(os.path.join(plot_path, 'inversion_columbia.pdf'),
                             dpi=150, bbox_inches='tight')

# figure 3 -------------
Plot_fig_3 = True

if Plot_fig_3:
    #Load McNabb et al. 2012 bed for the Columbia Glacier
    path_ele = os.path.join(MAIN_PATH, 'input_data/mcnabb.csv')
    thick07 = pd.read_csv(path_ele)
    thick07_h = thick07['bed07']
    farinotti = thick07['farinotti']
    model_1 = thick07['model_1']
    model_2 = thick07['model_2']
    model_3 = thick07['model_3']

    map_dx = gdir.grid.dx

    #print('grid',map_dx)
    inv = gdir.read_pickle('inversion_output', filesuffix='_without_calving_')[-1]
    fls = gdir.read_pickle('inversion_flowlines')

    inv_c = gdir.read_pickle('inversion_output')[-1]

    x = np.arange(fls[-1].nx) * fls[-1].dx * map_dx * 1e-3

    surface = fls[-1].surface_h

    thick = inv['thick']
    vol = inv['volume']
    bed = surface - thick

    thick_c = inv_c['thick']
    vol_c = inv_c['volume']
    bed_c = surface - thick_c

    bed_f = surface - farinotti
    bed_model_1 = surface - model_1
    bed_model_2 = surface - model_2
    bed_model_3 = surface - model_3

    # Plotting

    rcParams['axes.labelsize'] = 15
    rcParams['xtick.labelsize'] = 15
    rcParams['ytick.labelsize'] = 15

    rcParams['legend.fontsize'] = 12

    fig = plt.figure(figsize=(12,6))

    ax = fig.add_axes([0.07, 0.08, 0.7, 0.8])
    ax.plot(x, bed, color='grey', linewidth=2.5, label='Bed without frontal ablation')
    ax.plot(x, bed_c, color='k', linewidth=2.5, label = 'Bed with frontal ablation')
    ax.plot(x, surface, color='r', linewidth=2.5, label = 'Glacier surface')
    ax.plot(x, thick07_h, color='green', linestyle=':', linewidth=2.5,
            label='2007 bed (McNabb et al., 2012)')

    ax.plot(x, bed_f, color='orange', alpha=0.5, linewidth=2.5,
            label = 'Bed from the composite solution (Farinotti et al., 2019)')

    # ax.plot(x, bed_model_1, color='purple', alpha=0.5, linewidth=2.5,
    #         label='Huss and Farinotti, (2012)')
    #
    # ax.plot(x, bed_model_2, color='pink', alpha=0.5, linewidth=2.5,
    #         label='Frey et al. (2014)')
    #
    # ax.plot(x, bed_model_3, color='cyan', alpha=0.5, linewidth=2.5,
    #         label='Maussion et al. (2018)')


    ax.axhline(y=0, color='navy', linewidth=2.5, label= 'Sea level')
    ax.legend(loc='upper right')
    ax.set_xlabel('Distance along flowline [km]')
    ax.set_ylabel('Altitude [m]')
    letkm = dict(color='black', ha='left', va='top', fontsize=20,
                  bbox=dict(facecolor='white', edgecolor='black'))
    #ax.text(-10, 3500, 'c', **letkm)
    plt.tight_layout()
    #plt.show()
    plt.savefig(os.path.join(plot_path, 'columbia_profile.pdf'),
                             dpi=150, bbox_inches='tight')


#exit()

'''
    # # Computing spearman correlation because data is not normally distributed
    # rho, p = stats.spearmanr(thick07_h, bed)
    # rho_c, p_c = stats.spearmanr(thick07_h, bed_c)
    #
    # print('No calving and observations', rho, p)
    # print('With calving and observations', rho_c, p_c)
    #
    # alphacor = 0.05
    #
    # if p > alphacor:
    #     print('Samples are uncorrelated (fail to reject H0) p=%.5f', p)
    # else:
    #     print('Samples are correlated (reject H0) p=%.5f', p)
    #
    # if p_c > alphacor:
    #     print('Samples are uncorrelated (fail to reject H0) p=%.5f', p_c)
    # else:
    #     print('Samples are correlated (reject H0) p=%.5f', p_c)

    # calculating RMSD for oggm profiles and observations

    RMSD = oggm.utils.rmsd(thick07_h, bed)
    RMSD_c = oggm.utils.rmsd(thick07_h, bed_c)

    print('RMSD between observations and oggm with no calving',RMSD)
    print('RMSD between observations and oggm with calving', RMSD_c)

    mean_dev = oggm.utils.md(thick07_h, bed)
    mean_dev_c = oggm.utils.md(thick07_h, bed_c)
    print('mean difference between observations and oggm with no calving',mean_dev)
    print('mean difference between observations and oggm with calving', mean_dev_c)

    # Calculating the surface gradients
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, surface)

    print(slope, intercept)

    # Create a list of values in the best fit line
    abline_values = [slope * i + intercept for i in x]


    # calculate surface gradient ... surface is in m and space is now in m too
    obs = thick07_h - abline_values
    no_calving = bed - abline_values
    with_calving = bed_c - abline_values

    # Test for normality m
    k1, p1 = stats.normaltest(obs)
    k2, p2 = stats.normaltest(no_calving)
    k3, p3 = stats.normaltest(with_calving)

    alpha = 1e-3

    if p1 < alpha:
        print('sample is not normally distributed')
    else:
        print('sample is normally distributed')
    if p2 < alpha:
        print('sample is not normally distributed')
    else:
        print('sample is normally distributed')
    if p3 < alpha:
        print('sample is not normally distributed')
    else:
        print('sample is normally distributed')


    # Computing a spearman correlation test because not all the data is normally distributed
    rho, p = stats.pearsonr(obs, no_calving)
    rho_c, p_c = stats.pearsonr(obs, with_calving)

    print('No calving and observations', rho, p)
    print('With calving and observations', rho_c, p_c)

    alphacor = 0.05

    if p > alphacor:
        print('Samples are uncorrelated (fail to reject H0) p=%.5f', p)
    else:
        print('Samples are correlated (reject H0) p=%.5f', p)

    if p_c > alphacor:
        print('Samples are uncorrelated (fail to reject H0) p=%.5f', p_c)
    else:
        print('Samples are correlated (reject H0) p=%.5f', p_c)


    fig = plt.figure(figsize=(12,6))

    ax = fig.add_axes([0.07, 0.08, 0.7, 0.8])
    ax.plot(x, bed, color='grey', linewidth=2.5, label='Bed no calving')
    ax.plot(x, bed_c, color='k', linewidth=2.5, label = 'Bed with calving')
    ax.plot(x, surface, color='r', linewidth=2.5, label = 'Glacier surface')
    ax.plot(x, thick07_h, color='green', linestyle=':', linewidth=2.5,
            label='Observed bed 2007, (McNabb et al, 2012)')
    ax.plot(x, abline_values, color='r', linewidth=2.5, linestyle=':', label = 'Mean surface gradient')
    ax.axhline(y=0, color='navy', linewidth=2.5, label= 'Sea level')
    ax.legend(loc='upper right')
    ax.set_xlabel('Distance along flowline (Km)')
    ax.set_ylabel('Altitude (m)')
    letkm = dict(color='black', ha='left', va='top', fontsize=20,
                  bbox=dict(facecolor='white', edgecolor='black'))
    #ax.text(-10, 3500, 'c', **letkm)
    plt.tight_layout()
    #plt.show()
    plt.savefig(os.path.join(plot_path, 'columbia_profileplusgradient.png'),
                             dpi=150, bbox_inches='tight')
'''