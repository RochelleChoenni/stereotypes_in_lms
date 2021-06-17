
# Import packages
import os
import sys
import json
import spacy 
import matplotlib
import numpy as np 
import pandas as pd
import seaborn as sns
from nrclex import NRCLex
from itertools import groupby
import matplotlib.pyplot as plt
from collections import defaultdict

# Import files
sys.path.append("..")
from .target_dicts import eng_target_dict


def get_mlm_output(Model, topk, Group):
    names = {'BERT-B': 'bert-base-uncased', 'BERT-L': 'bert-large-uncased', 'RoBERTa-B': 'roberta-base', 'RoBERTa-L': 'roberta-large',
                'BART-B': 'bart-base', 'BART-L': 'bart-large', 'mBERT': 'bert-base-multilingual-uncased', 'XLMR-B': 'xlm-roberta-base', 'XLMR-L': 'xlm-roberta-large'}
    original_ranked=json.load(open('./mlm_output/'+names[Model]+'.json',"r")) 
    reranked=json.load(open('./mlm_output/'+names[Model]+'-corrected.json',"r")) 
    templates = []
    for i in original_ranked.keys():
        if Group.lower() in i.strip().split():
            prior = [i[0] for i in original_ranked[i]]
            post = [i[0] for i in reranked[i]]
            print(i)
            print('{:3s}{:20s} {:18s}'.format('', 'Original', 'Re-ranked'))
            for index, value in enumerate(prior[:topk], 1):
                print('{:2.0f} {:20s} {:20s}'.format(index, value, post[index-1]))
                    #print(index, value + '\t ' + post[index-1])
            print("\n")

def get_tgt(dataset='./data_collection/stereo_dataset/single_word_stereo_dataset.csv'):
    df = pd.read_csv(dataset,  encoding='utf-8', sep='\t')
    df = df.groupby('target_category')
    for i in df:
        targets = i[1].groupby('target_group').count()
        print('\033[1m'+i[0].capitalize()+'\033[0m', ':',', '.join([i.strip() for i in targets.index.values])+'\n')
        

    

def data_distr(dataset='./data_collection/stereo_dataset/single_word_stereo_dataset.csv'):
    df = pd.read_csv(dataset,  encoding='utf-8', sep='\t')
    df = df.groupby('search_engine').count()
    Engine = ['Google','Yahoo','Duckduckgo','Multiple']
    Num = df['input'].values
    dictionary = dict(zip(df.index.values, Num))
    Num = [dictionary[i.lower()] for i in Engine]
    New_Colors = ['blue','purple','brown','teal']
    plt.bar(Engine, Num, color=New_Colors)
    plt.title('Sample distribution over search engines', fontsize=12)
    plt.xlabel('Search engines', fontsize=12)
    plt.ylabel('Samples', fontsize=12)
    plt.grid(True)
    plt.show()

def search_engine_stereotypes(names):
    get_stereotypes_per_tgt(names)

def get_stereotypes_per_tgt(names, dataset='./data_collection/stereo_dataset/single_word_stereo_dataset.csv'):
    '''
    Retrieve all stereotypes from the search engine dataset corresponding to a target group
    , irrespective of the input query. 
    '''
    df = pd.read_csv(dataset,  encoding='utf-8', sep='\t')
    df = df.sort_values(by=['target_group'])
    targets, cats, stereoset = [], [], []
    prev = ''
    first= True
    for index, row in df.iterrows():
        group = row['target_group'].strip()
        cat = row['target_category'].strip()
        if group not in targets:
            if first:
                first= False
            else: 
                stereoset.append(new)
            new = []
            targets.append(group)
            cats.append(cat)

        inputs = row['input'].split()
        if inputs[-1] != 'so':
            ind = inputs.index('so')
            new.append(' '.join(inputs[ind+1:]) + " " +row['completion'].strip().lower())
        else:   
            new.append(row['completion'].strip().lower())

    stereo = list(zip(targets, cats, stereoset))
    stereo = sorted(stereo, key=lambda tup: tup[1])
    for i in list(set(cats)):
        tgt_stereo = [(item[0], item[2]) for item in stereo if item[1] == i]
        for tgt in tgt_stereo:
            if tgt[0] in names:
                print('{:25s} {:6s} {:5s}'.format('\033[1m'+tgt[0]+'\033[0m', '-->', ', '.join(list(set(tgt[1])))))

def plot_se_dist(df='data_collection/stereo_dataset/stereo_dataset.csv'):
    Engine = ['Google','Yahoo','duckduckgo','multiple']
    Num = [560,342,199,998]
    New_Colors = ['blue','purple','brown','teal']
    plt.bar(Engine, Num, color=New_Colors)
    plt.title('Sample distribution over search engines', fontsize=12)
    plt.xlabel('Search engines', fontsize=12)
    plt.ylabel('Samples', fontsize=12)
    plt.grid(True)
    plt.show()

def store_emotion_scores(targets, total, cat_of_interest, model_name, savedir):
    dump_dict = {}
    for i in range(0, len(targets)):
        dump_dict[targets[i]]=total[i]

    o = json.dumps(dump_dict)
    f = open(savedir+'/en_'+cat_of_interest+'_'+model_name+".json","w")
    f.write(o)
    f.close()


def get_stats(target_dict):
    print("Num cats: ", target_dict.keys())
    total = 0
    for k in target_dict.keys():
        print(k, len(target_dict[k]), len(list(set(target_dict[k]))))
        counter=collections.Counter(target_dict[k])
        print(counter)
        total += len(list(set(target_dict[k])))
    print("Total target groups: ", total)



def get_target_category(target, target_dict):
    keys = target_dict.keys()
    for k in keys:
        if target in target_dict[k]:
            return k

def accuracy_at_n(preds, target, n):
    if target in preds[:n]:
        return 1
    else:
        return 0



def plot_and_save_fig(data, targets, model_name, savedir, cat_of_interest, name, labels=['neg.', 'pos.',  'disgust', 'anger','sad', 'trust', 'joy', 'surp.', 'anticip.']):
    size = 15
    fig, ax = plt.subplots(figsize=(5,290))
    ax = sns.heatmap(np.array(data), vmin=10, vmax=65, cmap = sns.color_palette("mako", 100), center=38)
    ts = []
    for i in targets: 
        if i == 'religious people':
            ts.append('religionists ')
        else:
            ts.append(i)
    ax.set_xticks(np.arange(len(labels))+0.5)
    ax.set_yticks(np.arange(len(targets))+0.5)
    ax.set_xticklabels(labels)
    #ax.set_yticklabels(targets)
    ax.set_yticklabels(ts)

    plt.setp(ax.get_xticklabels(), rotation=90, fontsize=size, fontweight='bold')
    plt.setp(ax.get_yticklabels(), rotation=50, ha="right", rotation_mode="anchor", fontsize=size, fontweight='bold')
    ax.set_title(name +'-'+cat_of_interest, fontsize=size, fontweight='bold')

    fig.tight_layout()
    plt.show()
    if not os.path.exists(savedir+'/'+cat_of_interest):
    	os.makedirs(savedir+'/'+cat_of_interest)
    #fig.savefig(savedir+'/'+cat_of_interest+'/'+model_name+'_'+cat_of_interest+'.pdf', bbox_inches='tight')
    fig.savefig('paper_plots_finetuned/'+cat_of_interest+'/'+model_name+'_'+cat_of_interest+'.pdf', bbox_inches='tight',pad_inches = 0 )

    print(savedir+'/'+model_name+'_'+cat_of_interest+'.pdf', np.array(data).shape)
    plt.close()


def compare_model_predictions(json1, json2, key_list):
    dict1 = json.load(open('emotion_scores/'+json1,"r"))
    dict2 = json.load(open('emotion_scores/'+json2, "r"))
    
    for group in key_list:
        print("-----------", group)
     
        non_intersect = set(dict1[group][1])^set(dict2[group][1])
        breit = []
        base = []
        for i in non_intersect: 
            if i in dict1[group][1]:
                breit.append(i)
            elif i in dict2[group][1]:
                base.append(i)

        #print("Non overlap: ",len(list(set(dict1[group])^set(dict2[group])) ))
        print("Finetune: ", breit)
        print("Bert-BASE: ", base)
        print(len(breit), len(base))
    '''
    for group in sorted(dict1.keys(), key=lambda x:x.lower()):
        print(group)
        non_intersect = set(dict1[group])^set(dict2[group])
        breit = []
        base = []
        for i in non_intersect: 
            if i in dict1[group]:
                breit.append(i)
            elif i in dict2[group]:
                base.append(i)

        #print("Non overlap: ",len(list(set(dict1[group])^set(dict2[group])) ))
        print("BREIT: ", breit)
        print("Bert-BASE: ", base)
        print(len(breit), len(base))
    '''
'''
l1 = compare_model_predictions('race_en_fox_6.json', 'race_en_bert-base-uncased.json', ['blacks', 'black people', 'black kids', 'black parents'])
l2 = compare_model_predictions('country_en_fox_6.json', 'country_en_bert-base-uncased.json', ['Iraq'])

def find_articles_for_word(source, val_source, word):
    file1 = open(source, 'r')
    lines = file1.readlines()
    file2 = open(val_source, 'r')
    lines += file2.readlines()
    count =0
    for i in lines:
        if word in i:
            sents = i.split('.')
            #print(i)
            #for s in sents:
                #if word in s:
                    #print(s)
            count+=1
            #print('-------------------------')
    return count
'''
#w = ' elders '

#find_articles_for_word('archive/breitbart_train_split.txt','archive/breitbart_val_split.txt', w)
#find_articles_for_word('archive/guardian_train_split.txt', 'archive/guardian_val_split.txt', w)
#find_articles_for_word('archive/nytimes_train_split.txt', 'archive/nytimes_val_split.txt', w)

#find_articles_for_word('archive/cnn_train_split.txt','archive/cnn_val_split.txt', w)

#find_articles_for_word('archive/train_split.txt','archive/val_split.txt', w)