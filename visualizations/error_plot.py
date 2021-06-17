import numpy as np

import scipy as sp
import scipy.spatial
import scipy.stats
import logging
import json
import matplotlib.pyplot as plt

import pandas as pd
import os.path
from os import path

import seaborn as sns
import matplotlib.colors as mcolors
errors = [ 0.1512989 , 0.11659786, 0.06106059, 0.10633049, 0.16495416,   0.13815186, 0.17535678, 0.08261356, 0.14603082, 0.2209638,
  0.1287864 , 0.16199923, 0.11011358 ,0.16717786, 0.12883492,  0.12190544, 0.1362844 , 0.1089653,  0.11179641, 0.0793627]
print(len(errors))



df = pd.DataFrame({'Proportion of data': ['100%', '100%', '100%', '100%', '100%', '50%', '50%', '50%', '50%', '50%',
				'25%', '25%', '25%', '25%', '25%', '10%', '10%', '10%', '10%', '10%' ], 'Source':['New Yorker', 'Guardian', 'Reuters', 'FOX news','Breitbart']*4, 
				'Spearman Correlation':[  -0.42444444, -0.32777778, -0.57222222, -0.33222222, -0.33111111,   -0.41875, -0.435 ,  -0.385,   -0.4,  -0.43, 
				 -0.45125 , -0.3525 , -0.365,   -0.34625, -0.36625, -0.34125, -0.33375, -0.30625, -0.29375, -0.31625], 
				 'Errors':errors})
#ax = sns.pointplot('Proportion of data', 'Spearman Correlation', hue='Source',
#    data=df, dodge=True, join=True, ci=None)

dfCopy = df.copy()
duplicates = 30 # increase this number to increase precision
for index, row in df.iterrows():
    for times in range(duplicates):
        new_row = row.copy()
        new_row['Spearman Correlation'] = np.random.normal(row['Spearman Correlation'],row['Errors']) 
        dfCopy = dfCopy.append(new_row, ignore_index=True)


ax= sns.barplot(x="Proportion of data", y="Spearman Correlation", ci="sd", hue="Source", data=dfCopy, palette="Greens_d", capsize=0.1, errwidth=0.5)
print(df)
'''
# Find the x,y coordinates for each point
x_coords = [] 
y_coords = []
for point_pair in ax.collections:
    print(point_pair.get_offsets())
    for x, y in point_pair.get_offsets():
        x_coords.append(x)
        y_coords.append(y)

print(len(x_coords), len(y_coords))
'''

def errplot(x, y, yerr, **kwargs):
    ax = plt.gca()
    data = kwargs.pop("data")
    data.plot(x=x, y=y, yerr=yerr, kind="bar", ax=ax, **kwargs)

#g = sns.FacetGrid(df, col="Proportion of data", row="Spearman Correlation")
#g.map_dataframe(errplot, "Proportion of data", "Spearman Correlation", "Errors")
#colors = ['steelblue']*2 + ['coral']*2
#ax.errorbar(x_coords, y_coords, yerr=errors, fmt=' ', zorder=-1)
    #ecolor=colors, 
plt.gca().invert_yaxis()
plt.gca().invert_xaxis()
plt.savefig("Error-emotion-change.pdf")
plt.show()





'''
BERT-100: [ 0.         -0.42444444 -0.32777778 -0.57222222 -0.33222222 -0.33111111]
[0.         0.1512989  0.11659786 0.06106059 0.10633049 0.16495416]

BERT-50 :[ 0.      -0.41875 -0.435   -0.385   -0.4     -0.43   ]
[0.         0.13815186 0.17535678 0.08261356 0.14603082 0.2209638 ]



BERT-25: [ 0.      -0.45125 -0.3525  -0.365   -0.34625 -0.36625]
[0.         0.1287864  0.16199923 0.11011358 0.16717786 0.12883492]

BERT-10:
[ 0.      -0.34125 -0.33375 -0.30625 -0.29375 -0.31625]
[0.         0.12190544 0.1362844  0.1089653  0.11179641 0.0793627 ]


'''