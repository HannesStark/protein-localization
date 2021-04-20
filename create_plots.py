import math
import os

import h5py
from Bio import SeqIO
from tqdm import tqdm
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

sns.set_theme(style='whitegrid', rc={"xtick.bottom": True}, font_scale=0.9, font='Verdana')

plt.rcParams["figure.figsize"] = (8, 4)
plt.rcParams['figure.dpi'] = 300
font = {'family': 'normal',
        'weight': 'bold',
        'size': 10}

plt.rc('font', **font)
df = pd.read_csv('data/results/paper_tables.CSV')
remapping = {'Baseline': 'Baseline',
             'LocTree2': 'LocTree2',
             'MultiLoc2': 'MultiLoc2',
             'SherLoc2': 'SherLoc2',
             'Yloc': 'Yloc',
             'CELLO': 'CELLO',
             'iLoc-Euk': 'iLoc-Euk',
             'WoLF PSORT': 'WolF PSORT',
             'DeepLoc-BLOSUM62': 'DeepLoc62',
             'DeepLoc-profiles': 'DeepLoc',
             'AT SeqVec': 'AT SeqVec',
             'AT ProtBert': 'AT ProtBert',
             'AT ProtT5': 'AT ProtT5',
             'FFN SeqVec': 'MLP SeqVec',
             'FFN ProtBert': 'MLP ProtBert',
             'FFN ProtT5': 'MLP ProtT5',
             'LA SeqVec': 'LA SeqVec',
             'LA ProtBert': 'LA ProtBert',
             'LA ProtT5': 'LA ProtT5'}
df = df[['method', 'acc_deeploc', 'stdev_acc_deeploc', 'acc_hard', 'stdev_acc_hard']]
ordering = [0, 5, 2, 4, 6, 1, 7, 3, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
df['ordering'] = ordering
df = df.sort_values('ordering')
df['method'] = df['method'].map(remapping)

print(df)
# plt.rcParams.update({'font.size': 13})  # numbers font size
# plt.rcParams['xtick.labelsize'] = 14
#
# plt.rcParams['ytick.labelsize'] = 12

deep_loc, deep_loc_std = np.array(df['acc_deeploc']), np.array(df['stdev_acc_deeploc'])
hard_set, hard_std = np.array(df['acc_hard']), np.array(df['stdev_acc_hard'])
ind = np.arange(len(deep_loc))  # the x locations for the groups
width = 0.45
fig, ax = plt.subplots()
cmap = sns.color_palette("Greys_r", desat=.8)
cmap2 = sns.color_palette("muted", desat=1)
cmap3 = sns.color_palette("muted", desat=.6)
ticks = []
for i, val in enumerate(hard_set):
    if math.isnan(val) and False:
        print('true')
        ticks.append(ind[i])
    else:
        ticks.append(ind[i] - width / 2)
deep_rects = ax.bar(ticks, deep_loc, width, color=cmap[3], yerr=deep_loc_std, capsize=2, ecolor=cmap2[3],
                    error_kw={'elinewidth': 2}, label='setDeepLoc')
hard_rects = ax.bar(ind + width / 2, hard_set, width, color=cmap[0], yerr=hard_std, capsize=2, ecolor=cmap2[3],
                    error_kw={'elinewidth': 2}, label='setHard')
ax.margins(x=0.03)
ax.set_ylabel('10 state accuracy (Q10)')
ax.set_title('')
ax.set_ylim(0, 100)
ax.set_xticks(ind)
#ax.set_xticklabels(df['method'], rotation=90)
ax.set_xticklabels(df['method'], rotation=60, horizontalalignment='right')
ax.legend(loc='upper left')
plt.xticks(fontsize= 8)


def autolabel2(rects, displacement=[], displacement_width=[], xpos='center'):
    """
    Attach a text label above each bar in *rects*, displaying its height.

    *xpos* indicates which side to place the text w.r.t. the center of
    the bar. It can be one of the following {'center', 'right', 'left'}.
    """

    ha = {'center': 'center', 'right': 'left', 'left': 'right'}
    offset = {'center': 0, 'right': 1, 'left': -1}

    for i, rect in enumerate(rects):
        if i in displacement:
            placement = displacement_width[displacement.index(i)]
        else:
            placement = 0
        height = rect.get_height()
        ax.annotate('{:1.0f}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height + placement),
                    xytext=(offset[xpos] * 3, 3),  # use 3 points offset
                    textcoords="offset points",  # in both directions
                    ha=ha[xpos], va='bottom')




autolabel2(deep_rects, displacement=[16, 14, 12, 13], displacement_width=[1.5, 1.5, 5, 6], xpos="center")
autolabel2(hard_rects, displacement=[15, 14, 9], displacement_width=[1, 3, 3.5], xpos="center")

plt.hlines(77.4, 8.6, 18, colors=cmap[1], linestyles='dashed', label='')  # ,zorder=-1
plt.hlines(56.3, 8, 18, colors=cmap[1], linestyles='dashed', label='')
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

# from utils.general import LOCALIZATION
# fasta_path = 'data/embeddings/new_hard_set_t5_remapping.fasta'
# localizations = []
# for record in tqdm(SeqIO.parse(open(fasta_path), 'fasta')):
#    localization = record.description.split(' ')[2].split('-')[0]
#    localization = LOCALIZATION.index(localization)
#    localizations.append(localization)
#
# draws = 1000
# accuracies = []
# arr = np.copy(localizations)
# shuffle = np.array(localizations)
# for i in tqdm(range(draws)):
#    counter = 0
#    np.random.shuffle(shuffle)
#    accuracies.append((shuffle == arr).sum()/len(arr))
#
#
# print(100*np.array(accuracies).mean())
