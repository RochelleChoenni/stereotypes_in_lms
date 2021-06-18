import pandas as pd
import numpy as np
import collections 
import os 
import json


def get_train_test_split(df, source, save):

	df = df.loc[df['publication']==source]['content'][:int((4354/10))] #['article'][:4354]

	new_train = pd.DataFrame({'text': df[:int(np.round(df.shape[0]*0.9))].values})
	new_val = pd.DataFrame({'text': df[int(np.round(df.shape[0]*0.9)):].values})

	np.savetxt(r'archive/tenth_'+save+'_train_split.txt', new_train.values, fmt='%s')
	np.savetxt(r'archive/tenth_'+save+'_val_split.txt', new_val.values, fmt='%s')

def get_sources(csv_file):
	df = pd.read_csv(csv_file, encoding='utf-8', usecols = ['publication'], low_memory = True)
	elements_count = collections.Counter(df['publication'].values)
	for key, value in elements_count.items():
	   print(f"{key}: {value}")
	print("---------------------")


#get_sources('archive/articles1.csv')
#get_sources('archive/articles2.csv')
#get_sources('archive/articles3.csv')
#get_sources('archive/all-the-news-2-1.csv')
#df = pd.read_csv('archive/all-the-news-2-1.csv',  usecols = ['publication', 'article'], low_memory = True) #
#df = pd.read_csv('archive/all-the-news-2-1.csv', low_memory = True)

df= pd.read_csv('archive/articles1.csv', encoding='utf-8')
get_train_test_split(df, 'Breitbart', 'breitbart')

df= pd.read_csv('archive/articles2.csv', encoding='utf-8')
get_train_test_split(df, 'Fox News', 'foxnews')

df= pd.read_csv('archive/articles2.csv', encoding='utf-8')
get_train_test_split(df, 'Guardian', 'guardian')

df= pd.read_csv('archive/articles3.csv', encoding='utf-8')
get_train_test_split(df, 'Reuters', 'reuters')

