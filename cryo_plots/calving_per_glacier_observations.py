# This script will plot the calving flux of alaska
# with depth and width correction
import numpy as np
import pandas as pd
import os
import seaborn as sns
os.getcwd()
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.offsetbox import AnchoredText
import oggm

MAIN_PATH = os.path.expanduser('~/cryo_calving_2019/')

plot_path = os.path.join(MAIN_PATH, 'plots/')

lit_calving = os.path.join(MAIN_PATH,
                 'input_data/literature_calving_complete_plus_farinotti.csv')

run_output = os.path.join(MAIN_PATH,
    'output_data/5_runs_width_depth_correction/5_runs_width_depth_correction/')

filename1 = 'glacier_statistics_with_calving__cfgk_cfgA_cfgFS_no_correction.csv'
filename2 = 'glacier_statistics_with_calving__cfgk_cfgA_cfgFS_with_correction.csv'

glacier_char = os.path.join(run_output,
                            'run_alaska_MT_no_correction/'+filename1)

glacier_char_corrected = os.path.join(run_output,
        'run_alaska_MT_with_correction/'+filename2)


Fa_lit = pd.read_csv(lit_calving, index_col=0).sort_index(ascending=[True])

Fa_oggm = pd.read_csv(glacier_char, index_col=0).sort_index(ascending=[True])

Fa_oggm_corrected = pd.read_csv(glacier_char_corrected,
                                index_col=0).sort_index(ascending=[True])

Fa_oggm_sel = Fa_oggm.loc[Fa_lit.index]

Fa_oggm_sel_corrected = Fa_oggm_corrected.loc[Fa_lit.index]

d = {'McNabb et al. (2015)': Fa_lit['calving_flux_Gtyr']*1.091,
     'OGGM frontal ablation (default)': Fa_oggm_sel['calving_flux'],
     'OGGM frontal ablation (width and depth corrected)': Fa_oggm_sel_corrected['calving_flux'],
     'Farinotti et al. (2019)': Fa_lit['volume']*1e-9,
     'OGGM volume (default)': Fa_oggm_sel['inv_volume_km3'],
     'OGGM volume (width and depth corrected)': Fa_oggm_sel_corrected['inv_volume_km3']}

df = pd.DataFrame(data=d)

diff = df['OGGM frontal ablation (default)'] - df['McNabb et al. (2015)']
diff_corrected = df['OGGM frontal ablation (width and depth corrected)'] - df['McNabb et al. (2015)']




RMSD_no_correction = oggm.utils.rmsd(Fa_lit['calving_flux_Gtyr']*1.091,
                                     Fa_oggm_sel['calving_flux'])
RMSD_with_correction = oggm.utils.rmsd(Fa_lit['calving_flux_Gtyr']*1.091,
                                       Fa_oggm_sel_corrected['calving_flux'])

print('RMSD between observations and oggm no correction',RMSD_no_correction)
print('RMSD between observations and oggm with correction', RMSD_with_correction)

mean_dev = oggm.utils.md(Fa_lit['calving_flux_Gtyr']*1.091,
                                     Fa_oggm_sel['calving_flux'])
mean_dev_c = oggm.utils.md(Fa_lit['calving_flux_Gtyr']*1.091,
                                       Fa_oggm_sel_corrected['calving_flux'])

print('mean difference between observations and oggm with no correction', mean_dev)
print('mean difference between observations and oggm with correction', mean_dev_c)

diff = df['OGGM frontal ablation (default)'] - df['McNabb et al. (2015)']
diff_corrected = df['OGGM frontal ablation (width and depth corrected)'] - df['McNabb et al. (2015)']

# Set figure width and height in cm
width_cm = 12
height_cm = 6

fig = plt.figure(figsize=(width_cm, height_cm*2))
sns.set()
sns.set_color_codes("colorblind")
sns.set(style="white", context="talk")
# Plot settings
rcParams['axes.labelsize'] = 20
rcParams['xtick.labelsize'] = 16
rcParams['ytick.labelsize'] = 16
rcParams['legend.fontsize'] = 14

letkm = dict(color='black', ha='left', va='top', fontsize=20,
             bbox=dict(facecolor='white', edgecolor='black'))

N = len(df)
#ind = np.arange(0,2*N,2)
ind = np.arange(N)
#print(len(ind))
graph_width = 0.3
labels = df.index.values
#print(labels)

ax1 = fig.add_subplot(211)
ax1.tick_params(axis='both', bottom=True,
               left=True, width=2, direction='out', length=5)

p1 = plt.bar(ind, df['OGGM frontal ablation (default)'].values, graph_width,
             color = sns.xkcd_rgb["ocean blue"]),
             #edgecolor = sns.xkcd_rgb["ocean blue"])
p2 = plt.bar(ind+graph_width, df['OGGM frontal ablation (width and depth corrected)'].values, graph_width,
             color = sns.xkcd_rgb["teal green"])
             #edgecolor = sns.xkcd_rgb["teal green"])#, yerr=std_oggm)
p3 = plt.bar(ind+2*graph_width, df['McNabb et al. (2015)'].values, graph_width,
             color = sns.xkcd_rgb["burnt orange"])
             #edgecolor = sns.xkcd_rgb["burnt orange"] )#, yerr=std_fix)
ax1.axhline(y=0, color='k', linewidth=1.1)
ax1.tick_params(axis='both', bottom=True, left=True, width=2, direction='out', length=5)
plt.ylabel('Frontal ablation \n [km³$yr^{-1}$]')
plt.xticks(ind + graph_width, labels, rotation='vertical')
ax1.set_xticks(ind + graph_width)
ax1.set_xticklabels([])
plt.ylim(0,8)
plt.legend((p1[0], p2[0], p3[0]),
           ('OGGM default','OGGM width and depth corrected', 'McNabb et al. (2015)'), loc='upper right')
at = AnchoredText('a', prop=dict(size=20), frameon=True, loc=2)
ax1.add_artist(at)


ax2 = plt.subplot(212)
ax2.tick_params(axis='both',
                bottom=True, left=True, width=2, direction='out', length=5)

p4 = plt.bar(ind, df['OGGM volume (default)'].values, graph_width,
             color = sns.xkcd_rgb["ocean blue"]),
             #edgecolor = sns.xkcd_rgb["ocean blue"])
p5 = plt.bar(ind+graph_width, df['OGGM volume (width and depth corrected)'].values, graph_width,
             color = sns.xkcd_rgb["teal green"])
             #edgecolor = sns.xkcd_rgb["teal green"])#, yerr=std_oggm)
p6 = plt.bar(ind+2*graph_width, df['Farinotti et al. (2019)'].values, graph_width,
             color = sns.xkcd_rgb["burnt orange"])
             #edgecolor = sns.xkcd_rgb["burnt orange"] )#, yerr=std_fix)
ax2.axhline(y=0, color='k', linewidth=1.1)
ax2.tick_params(axis='both', bottom=True, left=True, width=2, direction='out', length=5)
plt.ylabel('Glacier volume \n [km³$yr^{-1}$]')
plt.xticks(ind + graph_width, labels, rotation='vertical')
ax2.set_xticks(ind + graph_width)
plt.legend((p4[0], p5[0], p6[0]),
           ('OGGM default','OGGM width and depth corrected',
            'Farinotti et al. (2019)'), loc='upper right')

at = AnchoredText('b', prop=dict(size=20), frameon=True, loc=2)
ax2.add_artist(at)

plt.margins(0.05)

#plt.show()
plt.savefig(os.path.join(plot_path, 'calving_per_glacier_draft.pdf'), dpi=150,
                  bbox_inches='tight')