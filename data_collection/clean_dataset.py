'''
This code is used to aggregate samples that are identical for multiple search engines,
and to clean up the duplicate predictions from the same search engines. Note that,
after running this code, further manual cleaning was performed. 
'''
import numpy as np
import collections
import pandas as pd


def filter_duplicates(dataframe, savefile):
    df = pd.read_csv(dataframe,  encoding='utf-8', sep='\t')
    df.loc[~df.index.duplicated(), :]
    index_names = df[ (df['target_category'] == 'country') & (df['input'].str.contains('are'))].index
    df.drop(index_names, inplace = True)
    index_names = df[ (df['target_category'] != 'country') & (df['input'].str.contains('is'))].index
    df.drop(index_names, inplace = True)
    df.dropna(inplace=True)
    df = df.loc[~df.index.duplicated(), :]
    sd = df.duplicated(subset=['input', 'target_category', 'target_group', 'completion'], keep=False)
    indices = [i for i, x in enumerate(sd.tolist()) if x == True]
    for i in indices:
        df.loc[i,'search_engine'] = 'multiple'
    df = df.drop_duplicates()
    df.to_csv(savefile, index=False,  encoding='utf-8', sep='\t')

