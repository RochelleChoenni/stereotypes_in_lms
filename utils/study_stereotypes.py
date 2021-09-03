import numpy as np

import scipy as sp
import scipy.spatial
import scipy.stats
import logging
import json
import matplotlib.pyplot as plt
from wordcloud import WordCloud

from .target_dicts import eng_target_dict
from utils.utils import plot_and_save_fig, get_target_category
import os.path

def detect_outliers(array, targets, m=3):
    order= ['Negative', 'Positive',  'Disgust', 'Anger','Sadness']
    outliers_dict = {}
    for emotion in range(0, len(order)):
        data = np.array([item[emotion] for item in array])
        med = np.median(data)
        d = np.abs(data - np.median(data))
        mdev = np.median(d)
        s = d / (mdev if mdev else 1.)
        indices = np.where(s >= m)
        indices = indices[0].tolist()
        outliers = []
        for i in indices:
            if data[i] > med:
                outliers.append((targets[i], data[i]-med))
        outliers_dict[order[emotion]] = outliers

    return outliers_dict


def plot_wordcloud(group, removed, new, same, name='wordcloud'):

    def color_func(word, *args, **kwargs):
        if word in removed:
            color = '#e01904' # red
        elif word in new:
            color = '#069c36' # green
        elif word in same:
            color = '#67746b' # blue
        else:
            color = '#000000' # black
        return color
  
    text = 'hello'
    words = [group] + removed + new + same
    text = {}
    for i in range(0,  len(words)):
        j = words[i]
        if i == 0:
            text[j] = 5
        else:
            text[j] =1
 
    wordcloud = WordCloud(background_color='white', min_font_size=12, color_func=color_func).generate_from_frequencies(text)

    plt.imshow(wordcloud, interpolation='bilinear')
    plt.tight_layout(pad=0)
    plt.axis("off")
    plt.show()

def get_shifts(interest, source='reuters'):
    pretrained_savedir = 'mlm_output/pretrained_dicts/roberta-base.json'
    finetuned_savedir='mlm_output/finetuned/roberta-base'
    pretrained_dictionary = json.load(open(pretrained_savedir,"r"))
    finetuned_dictionary = json.load(open(finetuned_savedir+'/'+source+'.json',"r"))

    for group in sorted(finetuned_dictionary.keys(), key=lambda x:x.lower()):     
        if group not in interest:
            continue;
        print("============ GROUP: ", group, '============')
        l_b = pretrained_dictionary[group]
        l_a = finetuned_dictionary[group]
        a_s = np.array_split(l_a, 5)

        b_s = np.array_split(l_b, 5)
        l_b = []
        for i in b_s:
            l_b += i[:15].tolist()
        l_b = sorted(set(l_b), key=l_b.index)

        l_a = []
        for i in a_s:
            l_a += i[:15].tolist()
        l_a = sorted(set(l_a), key=l_a.index)


        removed= [item for item in l_b if item not in l_a]
        new = [item for item in l_a if item not in l_b]
        same = [item for item in l_a if item in l_b]

     
        print("Removed: ", removed)
        print("Newly added: ", new)
        print("Remained: ", same)
        plot_wordcloud(group, removed, new, same)


def get_diff(cat_of_interest, modelA, modelB):
    pretrained_dictionary = json.load(open(modelA,"r"))

    finetuned_dictionary = json.load(open(modelB,"r"))
    for group in sorted(finetuned_dictionary.keys(), key=lambda x:x.lower()):     
        if get_target_category(group, eng_target_dict) != cat_of_interest:
            continue;
        print("============ GROUP: ", group, '============')
        l_b = pretrained_dictionary[group]
        l_a = finetuned_dictionary[group]
        a_s = np.array_split(l_a, 5)

        b_s = np.array_split(l_b, 5)
        l_b = []
        for i in b_s:
            l_b += i[:25].tolist()
        l_b = sorted(set(l_b), key=l_b.index)

        l_a = []
        for i in a_s:
            l_a += i[:25].tolist()
        l_a = sorted(set(l_a), key=l_a.index)


        removed= [item for item in l_b if item not in l_a]
        new = [item for item in l_a if item not in l_b]
        same = [item for item in l_a if item in l_b]

     
        print("Removed: ", removed)
        print("Newly added: ", new)
        print("Remained: ", same)
        