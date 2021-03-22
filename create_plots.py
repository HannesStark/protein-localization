import os

import h5py
from Bio import SeqIO
from tqdm import tqdm
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

sns.set_theme()
plt.rcParams["figure.figsize"] = (6, 8)
plt.rcParams['figure.dpi'] = 300
df = pd.read_csv('data/results/paper_tables.CSV')

df = df[['method', 'acc_deeploc', 'stdev_acc_deeploc', 'acc_hard', 'stdev_acc_hard']]

df['method'] = df['method'].map({'LocTree2': 'LocTree2',
                                 'MultiLoc2': 'MultiLoc2',
                                 'SherLoc2': 'SherLoc2',
                                 'Yloc': 'Yloc',
                                 'CELLO': 'CELLO',
                                 'iLoc-Euk': 'iLoc-Euk',
                                 'WoLF PSORT': 'WolF\nPSORT',
                                 'DeepLoc-BLOSUM62': 'DeepLoc\nBLOSUM',
                                 'DeepLoc-profiles': 'DeepLoc\nProfiles',
                                 'AT SeqVec': 'AT \nSeqVec',
                                 'AT ProtBert': 'AT \nProtBert',
                                 'AT ProtT5': 'AT \nProtT5',
                                 'FFN SeqVec': 'FFN \nSeqVec',
                                 'FFN ProtBert': 'FFN \nProtBert',
                                 'FFN ProtT5': 'FFN \nProtT5',
                                 'LA SeqVec': 'LA \nSeqVec',
                                 'LA ProtBert': 'LA \nProtBert',
                                 'LA ProtT5': 'LA \nProtT5'})

print(df)
plt.rcParams.update({'font.size': 13})  # numbers font size
plt.rcParams['xtick.labelsize'] = 14

plt.rcParams['ytick.labelsize'] = 12

deep_loc, deep_loc_std = np.array(df['acc_deeploc'])[::-1], np.array(df['stdev_acc_deeploc'])[::-1]
hard_set, hard_std = np.array(df['acc_hard'])[::-1], np.array(df['stdev_acc_hard'])[::-1]
ind = np.arange(len(deep_loc))  # the x locations for the groups
width = 0.45
fig, ax = plt.subplots()
cmap = sns.color_palette("Greys_r", desat=.8)
cmap2 = sns.color_palette("muted", desat=1)
cmap3 = sns.color_palette("muted", desat=.6)
deep_rects = ax.barh(ind - width / 2, deep_loc, width, color=cmap[0], xerr=deep_loc_std, capsize=2, ecolor=cmap2[3],error_kw={'elinewidth':2},label='setDeepLoc')
hard_rects = ax.barh(ind + width / 2, hard_set, width, color=cmap[4], xerr=hard_std, capsize=2, ecolor=cmap2[3],error_kw={'elinewidth':2}, label='setHard')
ax.margins(y=0)
ax.set_ylabel('')
ax.set_title('')
ax.set_xlim(40, 100)
ax.set_yticks(ind)
ax.set_yticklabels(df['method'])
ax.legend()


def autolabel(rects, displacement=[], displacement_width=[], xpos='center'):
    """
    Attach a text label above each bar in *rects*, displaying its height.
    *xpos* indicates which side to place the text w.r.t. the center of
    the bar. It can be one of the following {'center', 'right', 'left'}.
    """

    ha = {'center': 'center', 'right': 'left', 'left': 'right'}
    offset = {'center': 0, 'right': 1, 'left': -1}

    for i, rect in enumerate(rects):
        width = rect.get_width()  # get bar length
        if i in displacement:
            placement = displacement_width[displacement.index(i)]
        else:
            placement = 1
        ax.text(width + placement,  # set the text at 1 unit right of the bar
                rect.get_y() + rect.get_height() / 2,  # get Y coordinate + X coordinate / 2
                '{:1.0f}'.format(width),  # set variable to display, 2 decimals
                ha='left',  # horizontal alignment
                va='center')  # vertical alignment


autolabel(deep_rects, displacement=[5], displacement_width=[2], xpos="left")
autolabel(hard_rects, displacement=[0, 1, 3, 4], displacement_width=[2, 3, 2, 2], xpos="right")
plt.vlines(60.92, 0, 10, colors=cmap[1], linestyles='dashed', label='')  # ,zorder=-1
fig.tight_layout()

plt.show()
plt.clf()

raise Exception

f, ax = plt.subplots(figsize=(6, 15))
df = df.melt(id_vars=['method', 'stdev_acc_deeploc', 'stdev_acc_hard'])
g = sns.barplot(data=df, x='value', y='method', hue='variable', palette='gray')
for p in ax.patches:
    width = p.get_width()  # get bar length
    ax.text(width + 1,  # set the text at 1 unit right of the bar
            p.get_y() + p.get_height() / 2,  # get Y coordinate + X coordinate / 2
            '{:1.2f}'.format(width),  # set variable to display, 2 decimals
            ha='left',  # horizontal alignment
            va='center')  # vertical alignment
# for index, row in df.iterrows():
#    g.text(row.name,row.value, round(row.value,2), color='black', ha="center")
plt.errorbar(x=df['value'], y=df['method'], xerr=df['stdev_acc_deeploc'], fmt='none', c='black', capsize=3)
plt.xlim(40, 100)
plt.ylabel('')
plt.tight_layout()
plt.show()