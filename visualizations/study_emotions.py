import numpy as np
import logging
import json
import matplotlib.pyplot as plt
import pandas as pd
import os.path
from os import path
import seaborn as sns
from  matplotlib.colors import LinearSegmentedColormap
import sys
sys.path.append("..")


def cutstom_map():
    c = ['black', 'dimgrey', 'grey', 'silver', 'green', 'darkgreen', 'gold', 'orange', 'darkorange', 'orangered', "red", 'firebrick', "maroon", 'saddlebrown']
    v = [0, .1, .15, 0.2, .35, .4, .5, .6,0.7, 0.75, .8, .85, 0.9, 1.]
    l = list(zip(v,c))
    cmap=LinearSegmentedColormap.from_list('rg',l, N=256)
    return  cmap

def plot_and_save_fig(data, targets, model_name, labels=['neg.', 'pos.', 'disgust', 'anger', 'fear', 'sad', 'trust', 'joy']):
    size = 15
    fig, ax = plt.subplots(figsize=(5,7))
    ndata = []
    ndata = [i[:8] for i in data]
    cmap = cutstom_map()
    ax = sns.heatmap(np.array(ndata), vmin=10, vmax=65, cmap = cmap) 
    ts = []
    for i in targets: 
        if i == 'religious people':
            ts.append('religionists ')
        else:
            ts.append(i)
    ax.set_xticks(np.arange(len(labels))+0.5)
    ax.set_yticks(np.arange(len(targets))+0.5)
    ax.set_xticklabels(labels)
    ax.set_yticklabels(ts)

    plt.setp(ax.get_xticklabels(), rotation=90, fontsize=size, fontweight='bold')
    plt.setp(ax.get_yticklabels(), rotation=50, ha="right", rotation_mode="anchor", fontsize=size, fontweight='bold')

    fig.tight_layout()
    plt.show()


def plot_emotion_vectors(Model, Groups):
    names = {'BERT-B': 'bert-base-uncased', 'BERT-L': 'bert-large-uncased', 'RoBERTa-B': 'roberta-base', 'RoBERTa-L': 'roberta-large',
                'BART-B': 'bart-base', 'BART-L': 'bart-large', 'mBERT': 'bert-base-multilingual-uncased', 'XLMR-B': 'xlm-roberta-base', 'XLMR-L': 'xlm-roberta-large'}
    if path.isfile('./aggregate_emotion_scores/'+Model+'.json'):
        dictionary = json.load(open('./aggregate_emotion_scores/'+Model+'.json',"r")) 
        targets = []
        total = []
        for i in dictionary.keys(): 
            if i in Groups:
                targets.append(i)
                total.append(dictionary[i][0])
    else: 
  
        dictionary = json.load(open('mlm_output/'+names[Model]+'.json',"r"))
        total = []
        counts = []
        df = pd.read_excel('emotion_scores/NRC-Emotion-Lexicon-v0.92-In105Languages-Nov2017Translations.xlsx',  encoding='utf-8', sep='\t')
        targets = []
        word_list = []
        print(Groups)
        G = Groups.split(',')
        for group in G:     
            print("--------------"+ group + "-------------------")
            anger_list = []
            disgust_list = []
            fear_list = []
            sad_list = []
            emo_dict = {'Negative':0, 'Positive':0,  'Disgust':0, 'Anger':0, 'Fear':0, 'Sadness':0, 'Trust':0, 'Joy':0, 'Surprise':0, 'Anticipation':0}
            w_dict_g= list(set(dictionary[group]))

            counter = 0
            covered=[]
            for w in w_dict_g:
                w=w.strip()
                try: 
                    row = df.loc[df['English (en)'].str.lower() == w.lower()]
                    for emotie in emo_dict.keys():  
                        if row[emotie].values[0] == 1:
                            emo_dict[emotie]+=1
                            if emotie == 'Negative':
                                anger_list.append(w)
                            elif emotie == 'Positive':
                                disgust_list.append(w)
                            elif emotie == 'Joy':
                                fear_list.append(w)
                            elif emotie == 'Trust':
                                sad_list.append(w)
                    counter+=1
                    covered.append(w)
                except:
                    continue;

            order= ['Negative', 'Positive',  'Disgust', 'Anger','Fear','Sadness', 'Trust', 'Joy', 'Surprise', 'Anticipation']
            x = [np.round(emo_dict[o],2) for o in order]
            targets.append(group)
            total.append(x)
            counts.append(counter)
            word_list.append(covered)
           

        total = [[(l/counts[n])*100 for l in total[n]] for n in range(0, len(total))]

        save_dict = {}
        for i in range(0, len(targets)):
            save_dict[targets[i]] = [total[i], word_list[i]]

        a_file = open(savedir+'/'+name+'.json', "w")

        json.dump(save_dict, a_file)

        a_file.close()

    plot_and_save_fig(total, targets, names[Model])

   
    return 


def compute_diff(a, b, c):
    diff_a = list(set(a)-set(b)-set(c))
    diff_b = list(set(b)-set(a)-set(c))
    diff_c = list(set(c)-set(a)-set(b))
    print('religion: ', diff_a)
    print("homo: ", diff_b)
    print('liberal', diff_c)

def cd(a, b):
    diff_a = list(set(a)-set(b))
    diff_b = list(set(b)-set(a))
    print('roberta: ', diff_a)
    print("bart: ", diff_b)
