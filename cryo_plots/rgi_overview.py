import os
import salem
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns
import pandas as pd
import geopandas as gp
os.getcwd()
from cartopy.feature import ShapelyFeature
from cartopy.io.shapereader import Reader
import cartopy.crs as ccrs
import geopandas as gpd
from collections import OrderedDict
sns.set_context('poster')
sns.set_style('whitegrid')
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER


MAIN_PATH = os.path.expanduser('~/cryo_calving_2019/')

plot_path = os.path.join(MAIN_PATH, 'plots/')

RGI_FILE = os.path.join(MAIN_PATH,'input_data/01_rgi60_Alaska_modify/01_rgi60_Alaska.shp')
rgi_sub_regions = os.path.join(MAIN_PATH,'input_data/00_rgi50_regions/00_rgi50_O2Regions.shp')

filename_coastline = os.path.join(MAIN_PATH,'input_data/ne_10m_coastline/ne_10m_coastline.shp')

mc_nabb = os.path.join(MAIN_PATH,'input_data/Mcnabb_glaciers/01_rgi50_Alaska.shp')
rgi_sub_regions

# Reading all the shapefiles

#RGI sub-regions
dg = gpd.read_file(rgi_sub_regions)

#Read hi resolution coast line
shape_coast = ShapelyFeature(Reader(filename_coastline).geometries(),
                             ccrs.PlateCarree(), facecolor='none',
                            edgecolor='black')

#Read McNabb glaciers
dm = gpd.read_file(mc_nabb)
dm['centroid_column'] = dm.centroid
dm = dm.set_geometry('centroid_column')

#RGI v6
df = gpd.read_file(RGI_FILE)
df.set_index('RGIId')
index = df.index.values

#Classify the glaciers
sub_mar = df[df['TermType'].isin([1]) ]
sub_lake = df[df['TermType'].isin([2])]
sub_land = df[df['TermType'].isin([0])]

#Get glacier's coordinates
sub_mar['centroid_column'] = sub_mar.centroid
sub_mar = sub_mar.set_geometry('centroid_column')
sub_lake['centroid_column'] = sub_lake.centroid
sub_lake = sub_lake.set_geometry('centroid_column')

sub_land['centroid_column'] = sub_land.centroid
sub_land = sub_land.set_geometry('centroid_column')

f = plt.figure(figsize=(20, 18))

from matplotlib import rcParams

rcParams['axes.labelsize'] = 25
rcParams['xtick.labelsize'] = 25
rcParams['ytick.labelsize'] = 25
rcParams['legend.fontsize'] = 25

ax = plt.axes(projection=ccrs.Mercator())

sns.set_style("white")

# mark a known place to help us geo-locate ourselves
ax.set_extent([-157, -129, 55, 64.5], crs=ccrs.PlateCarree())


#sns.set_color_codes("colorblind")

dg.plot(ax=ax, transform=ccrs.PlateCarree(),
        edgecolor='k', facecolor='w', alpha=0.5);
dg.plot(ax=ax, transform=ccrs.PlateCarree(),
        edgecolor='k', facecolor='none', linewidth=2, alpha=0.5);

ax.add_feature(shape_coast, linewidth=2)


dm.plot(ax=ax, transform=ccrs.PlateCarree(),
        label='Frontal ablation estimates \n (McNabb et al., 2015)',
        color='orange',marker='o', markersize=400)
sub_mar.plot(ax=ax, transform=ccrs.PlateCarree(),
        label='Marine terminating glaciers',
        color='blue',marker='.', markersize=300, alpha=1)
sub_lake.plot(ax=ax, transform=ccrs.PlateCarree(),
        label='Lake terminating glaciers',
        color='olive',marker='.', markersize=300, alpha=1)
sub_land.plot(ax=ax, transform=ccrs.PlateCarree(),
        label='Land terminating glaciers',
        color='gray', marker='.', markersize=30, alpha=0.1)

ax.legend(scatterpoints = 1, loc='best')

handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys(), #fontsize=15,
           bbox_to_anchor=(0.2, 0.22), loc=2, borderaxespad=0., fancybox=True)

gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=2, color='gray', alpha=0.5, linestyle='--')
gl.xlabels_bottom = False
gl.ylabels_right = False
gl.xlines = False
gl.ylines = False
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER
gl.xlabel_style = {'size': 25}
gl.ylabel_style = {'size': 25}



ax.text(-155.45,64.2,'1', transform=ccrs.PlateCarree(),
            ha='left', va='bottom', fontsize=25,
            bbox=dict(facecolor='w', edgecolor='k', alpha=0.7))

ax.text(-155.45,63.0,'2', transform=ccrs.PlateCarree(),
            ha='left', va='bottom', fontsize=25,
            bbox=dict(facecolor='w', edgecolor='k', alpha=0.7))

ax.text(-154.49,55.74,'3', transform=ccrs.PlateCarree(),
            ha='left', va='bottom', fontsize=25,
            bbox=dict(facecolor='w', edgecolor='k', alpha=0.7))

ax.text(-148.79,57.95,'4', transform=ccrs.PlateCarree(),
            ha='left', va='bottom', fontsize=25,
            bbox=dict(facecolor='w', edgecolor='k', alpha=0.7))

ax.text(-141.49,57.87,'5', transform=ccrs.PlateCarree(),
            ha='left', va='bottom', fontsize=25,
            bbox=dict(facecolor='w', edgecolor='k', alpha=0.7))

ax.text(-138.61,55.16,'6', transform=ccrs.PlateCarree(),
            ha='left', va='bottom', fontsize=25,
            bbox=dict(facecolor='w', edgecolor='k', alpha=0.7))

plt.tight_layout()
#plt.show()
plt.savefig(os.path.join(plot_path, 'rgi_overview.pdf'), bbox_inches='tight')

