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
    #wordcloud = WordCloud(width = 800, height = 800,
    #            background_color ='white',
    #            stopwords = stopwords,
    #            min_font_size = 10).generate(comment_words)

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
        #if i == 10:
        #    break;
    
    #text = [x.encode('latin-1') for x in text]
    wordcloud = WordCloud(background_color='white', min_font_size=12, color_func=color_func).generate_from_frequencies(text)
    #wordcloud.to_file(name+'.pdf') 
    # Display the generated image:
    # the matplotlib way:
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
        

''' 
cat = 'profession'
interest = ['researchers']
get_shifts(cat, interest ,source='reuters')
get_shifts(cat, interest,source='guardian')

get_shifts(cat, interest,source='new_yorker')
get_shifts(cat, interest, source='breitbart')

get_shifts(cat, interest,source='fox')
'''
#get_shifts('profession', interest= ['police officers'],source='reuters')
#get_shifts('profession', interest= ['police officers'],source='guardian')

#get_shifts('race', interest= ['asians', 'black men'],source='new_yorker')
#get_shifts('race', interest= ['asians', 'black men'],source='breitbart')

#get_shifts('profession', interest= ['police officers'],source='breitbart')


#get_diff('religion', 'new_result/roberta-base.json',  'new_result/roberta-large.json' )
#group, removed, new, same = get_shifts('religion')
#plot_wordcloud('capitalists', ['arrogant', 'desperate', 'angry', 'interesting', 'kind', 'secretive', 'ruthless', 'mad'], ['ambitious', 'aggressive', 'optimistic'],
 #['stupid', 'lucky', 'rich', 'powerful', 'cruel', 'smart', 'corrupt', 'naive', 'weak', 'wealthy', 'successful', 'happy', 'valuable', 'cynical', 'clever',  'greedy', 'cautious', 'selfish', 'stubborn'])

#plot_wordcloud('capitalists', ['lucky', 'greedy', 'naive', 'poor', 'corrupt', 'desperate', 'angry', 'valuable', 'interesting', 'kind', 'cynical', 'selfish', 'generous', 'secretive', 'ruthless', 'weak'],
# ['big', 'bold', 'intelligent', 'fast', 'cheap', 'bad', 'nervous', 'crazy', 'exciting', 'flexible', 'optimistic', 'modest', 'unpredictable', 'eager', 'rude', 'worried']
# ,['smart', 'stupid', 'rich', 'powerful', 'predictable', 'strong', 'wealthy', 'cruel', 'happy', 'mad', 'dangerous', 'clever', 'successful', 'cautious', 'stubborn', 'arrogant'])

'''
plot_wordcloud('Trump supporters', ['vocal', 'interested', 'lucky', 'close', 'friendly', 'paranoid', 'stubborn', 'cautious', 'silent', 'busy', 'divided'], 
    ['enthusiastic', 'different', 'proud', 'confused', 'violent', 'mean'], 
     ['angry', 'mad', 'upset', 'excited', 'afraid', 'eager', 'desperate', 'small', 'worried', 'crazy', 'hostile', 'happy', 'stupid', 'strong', 'powerful', 'bold', 'special', 'popular', 'nervous', 'unhappy', 'kind', 'smart', 'young', 'conservative', 'impatient', 'quiet', 'surprised', 'scared'] 
     , 'trump_breitbart')

plot_wordcloud('Trump supporters',  ['vocal', 'interested', 'close', 'friendly', 'paranoid', 'stubborn', 'cautious', 'silent']
,['enthusiastic', 'confident', 'confused', 'secretive', 'tense']
, ['angry', 'mad', 'upset', 'small', 'desperate', 'stupid', 'eager', 'strong', 'hostile', 'powerful', 'divided', 'worried', 'afraid', 'excited', 'nervous', 'happy', 'unhappy', 'bold', 'crazy', 'special', 'popular', 'smart', 'kind', 'conservative', 'young', 'lucky', 'impatient', 'quiet', 'busy', 'surprised', 'scared']
,'trump_fox')

plot_wordcloud('Trump supporters', ['small', 'eager', 'unhappy', 'vocal', 'crazy', 'close', 'friendly', 'stubborn', 'silent', 'busy']
, ['enthusiastic', 'different', 'proud', 'loyal', 'confused', 'suspicious', 'concerned']
,  ['angry', 'mad', 'afraid', 'worried', 'divided', 'powerful', 'upset', 'excited', 'scared', 'nervous', 'strong', 'hostile', 'desperate', 'surprised', 'happy', 'popular', 'bold', 'interested', 'special', 'smart', 'kind', 'conservative', 'lucky', 'stupid', 'young', 'cautious', 'quiet', 'paranoid', 'impatient']
, 'trump_guardian')


plot_wordcloud('Trump supporters',  ['stupid', 'crazy', 'special', 'close', 'stubborn', 'silent', 'busy', 'surprised', 'scared']
, ['large', 'big', 'many', 'restless', 'loyal', 'enthusiastic', 'confident', 'generous', 'aggressive', 'passive', 'poor']
, ['small', 'angry', 'young', 'eager', 'strong', 'afraid', 'bold', 'powerful', 'mad', 'desperate', 'divided', 'happy', 'excited', 'unhappy', 'popular', 'interested', 'upset', 'vocal', 'nervous', 'lucky', 'smart', 'conservative', 'friendly', 'kind', 'hostile', 'impatient', 'paranoid', 'cautious', 'quiet', 'worried']
, 'trump_newyorker')
'''

'''
plot_wordcloud('police officers', [' successful', ' nice', ' rude', ' slow', ' black'],
 [' cold', ' deadly', ' inept', ' unreliable', ' reactive'],
[' violent', ' afraid', ' dangerous', ' scared', ' racist', ' bad', ' fearful', ' aggressive', ' paranoid', ' important', ' angry', ' controversial', ' nervous', ' mean', ' special', ' effective', ' good', ' great', ' scary', ' tough', ' valuable', ' different', ' smart', ' lazy', ' stupid', ' corrupt', ' incompetent', ' popular', ' dumb', ' quiet', ' white']
 , 'police_ny')

plot_wordcloud('police officers',  [' popular', ' controversial', ' scary', ' successful', ' corrupt', ' mean', ' black'],
 [' reactive', ' expensive', ' exceptional', ' difficult', ' inept', ' loyal', ' timid', ' cautious', ' polite'],
[' afraid', ' violent', ' dangerous', ' scared', ' paranoid', ' fearful', ' nervous', ' racist', ' angry', ' aggressive', ' important', ' quiet', ' bad', ' special', ' effective', ' good', ' great', ' smart', ' tough', ' valuable', ' different', ' stupid', ' incompetent', ' lazy', ' slow', ' nice', ' dumb', ' rude', ' white']
, 'police_fox')
'''
'''
plot_wordcloud('black people',  [' mad', ' dangerous', ' ignorant'],
 [' miserable', ' good', ' beautiful', ' interesting', ' happy', ' successful', ' important', ' lucky', ' dead'],
[' angry', ' afraid', ' hated', ' oppressed', ' poor', ' racist', ' bad', ' different', ' scared', ' violent', ' silent', ' unhappy', ' white', ' stupid', ' special', ' great', ' powerful', ' strong', ' misunderstood', ' lazy', ' quiet', ' ugly']
    , 'black_ny')


plot_wordcloud('black people', [' powerful', ' misunderstood'],
[' upset', ' bitter', ' anxious', ' uncomfortable', ' proud', ' stubborn', ' negative', ' rude'],
[' angry', ' afraid', ' racist', ' silent', ' violent', ' scared', ' different', ' oppressed', ' poor', ' bad', ' quiet', ' stupid', ' hated', ' special', ' unhappy', ' strong', ' mad', ' dangerous', ' great', ' lazy', ' white', ' ignorant', ' black', ' ugly']
, 'black_fox')
'''
params = 'within_eng'

if params == 'de':
    models = ['de_bert-base-multilingual-uncased.json', 'de_xlm-roberta-base.json', 'de_bert-base-german-cased.json']
    output_dir = 'rsa_de'
    labels = ['de_mbert', 'de_xlmr', 'de_bert']
    src_lang = 'German (de)'
elif params == 'eng':
    models = ['bert-large-uncased.json', 'roberta-large.json', 'bart-large.json', 'bert-base-multilingual-uncased.json', 'xlm-roberta-base.json']
    labels = ['bert', 'roberta', 'bart', 'mbert', 'xlmr']
    output_dir = 'rsa_en'
elif params == 'spa':
    models = ['es_bert-base-multilingual-uncased.json', 'es_xlm-roberta-base.json', 'es_beto.json']
    labels = ['es_mbert', 'es_xlmr', 'beto']
    output_dir = 'rsa_es'
elif params == 'nl':
    models = ['nl_bert-base-multilingual-uncased.json', 'nl_xlm-roberta-base.json', 'nl_bert-base-dutch-cased.json']
    output_dir = 'rsa_nl'
    labels = ['nl_mbert', 'nl_xlmr', 'nl_bert']
    src_lang = 'Dutch (nl)'
elif params == 'within_eng':
    models = ['en_bert-base-uncased.json', 'en_fox_6.json', 'en_guardian_6.json', 'en_nytimes_6.json', 'en_breitbart_6.json', 'en_cnn_6.json']
    labels = ['bert-base', 'fox', 'guardian', 'nytimes', 'breibart', 'cnn']
    output_dir = 'rsa_news_freq'    

'''
for cat_of_interest in ['country', 'race']:#, 'gender', 'religion', 'country', 'age', 'political', 'profession', 'lifestyle', 'sexuality']: 
    for i in models:
        print(i)
        vectors = []
        words = []
        with open("emotion_scores/"+cat_of_interest+"_"+i, "r") as file:
            dictionary = json.load(file)
            for t in dictionary.keys():
                vectors.append(dictionary[t][0])
                words.append(dictionary[t][1])
            outlier_dict = detect_outliers(vectors, list(dictionary.keys()))
        if i != 'en_bert-base-uncased.json':
            order= ['Negative', 'Positive',  'Disgust', 'Anger','Sadness']
            for emotion in order:
                outliers = outlier_dict[emotion]
                print(emotion)
                for w in outliers:
                    name = i.replace('_6.json', '')
                    count =  find_articles_for_word('archive/'+name[3:]+'_train_split.txt', 'archive/'+name[3:]+'_val_split.txt', w[0])
                    print(w[0], " occurences ", count, ' diff ', w[1])
'''

