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
    '''
    for i in data:
        j = i[0:2]+i[4:]
        ndata.append(j)
    '''
    ndata = [i[:8] for i in data]
    cmap = cutstom_map()
    ax = sns.heatmap(np.array(ndata), vmin=10, vmax=65, cmap = cmap) #'sns.color_palette("mako", 100), center=38)
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
    #ax.set_title(name , fontsize=size, fontweight='bold')

    fig.tight_layout()
    plt.show()
    #fig.savefig(savedir+'/'+name+'.pdf', bbox_inches='tight',pad_inches = 0 )

    #plt.close()

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
           

            #print(group, "Found: ", counter, "Not found: ", len(w_dict_g)-counter, "%: ", (counter/ len(w_dict_g)) *100)

        #total  = [[(x-min(l))/(max(l)-min(l))*100 for x in l] for l in total]
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

#g = ['religious people', 'homosexuals', 'liberals', 'black people', 'white people', 'scots', 'Puerto Rico', 'Greece', 'strippers', 'husbands', 'poor people', 'teenagers'] 
#g = ['christians', 'trump supporters', 'conservatives', 'celebrities', 'journalists', 'academics', 'gay people', 'Iraq', 'wives', 'ladies', 'black women', 'teenagers'] 
#g = ['christians', 'trump supporters', 'conservatives', 'celebrities', 'lawyers', 'police officers', 'academics', 'gay people', 'Iraq', 'asians', 'ladies', 'black men', 'teenagers', 'elders'] 
#g = ['christians',  'conservatives', 'celebrities', 'police officers', 'academics', 'gay people', 'Iraq', 'trump supporters', 'asians', 'ladies', 'black men', 'teenagers'] 

#g = ['christians', 'homosexuals', 'immigrants', 'venezualans', 'Singapore', 'business men', 'ladies', 'goths', 'old people']
#plot_emotion_vectors('new_result/bert-base-multilingual-uncased', g,
#                    'paper_plots', 'mBERT')
'''
en_target_emotions('finetuned1epoch/roberta-base/fox', g,
                     'paper_plots', 'FoxNews')
en_target_emotions('finetuned1epoch/roberta-base/new_yorker', g,
                     'paper_plots', 'NewYorker')
en_target_emotions('finetuned1epoch/roberta-base/breitbart', g,
                     'paper_plots', 'Breitbart')
en_target_emotions('finetuned1epoch/roberta-base/reuters', g,
                     'paper_plots', 'Reuters')
en_target_emotions('finetuned1epoch/roberta-base/guardian', g,
                     'paper_plots', 'Guardian')

'''
# ---------------- ROBERTA -------------
anger_rel = ['immoral', 'ridiculous', 'brutal', 'bad', 'intense', 'annoying', 'fierce', 'terrible', 'crazy', 'dishonest', 'obnoxious', 'opposed', 'aggressive', 'cruel', 'disappointed', 'guilty', 'hateful', 'ferocious', 'honest', 'worthless', 'barbaric', 'arrogant', 'nasty', 'insane', 'incompetent', 'disliked', 'offensive', 'offended', 'ill', 'angry', 'miserable', 'horrible', 'insecure', 'hate', 'ruthless', 'frustrated', 'inept', 'powerful', 'defiant', 'noisy', 'upset', 'perverse', 'disgusting', 'violent', 'evil', 'hostile', 'delusional', 'troublesome', 'suspicious', 'destructive', 'vicious', 'disrespectful', 'hot', 'murderous', 'furious', 'despicable', 'awful', 'disturbed', 'unhappy', 'depressed', 'mad', 'selfish', 'controversial', 'oppressive']
disgust_rel = ['immoral', 'ridiculous', 'sick', 'bad', 'intense', 'fierce', 'pious', 'terrible', 'dishonest', 'obnoxious', 'cruel', 'ugly', 'disappointed', 'hateful', 'ferocious', 'interested', 'honest', 'worthless', 'barbaric', 'toxic', 'misguided', 'arrogant', 'nasty', 'weird', 'abnormal', 'offensive', 'ill', 'angry', 'miserable', 'horrible', 'hate', 'ruthless', 'irrational', 'unpopular', 'inept', 'powerful', 'perverse', 'disgusting', 'fat', 'violent', 'uneasy', 'evil', 'hostile', 'hypocritical', 'destructive', 'vicious', 'disrespectful', 'murderous', 'furious', 'despicable', 'ignorant', 'awful', 'shameless', 'unpleasant', 'unhappy', 'mad', 'selfish', 'greedy', 'oppressive']
fear_rel = ['serious', 'immoral', 'frightened', 'brutal', 'bad', 'intense', 'fierce', 'terrible', 'crazy', 'opposed', 'loyal', 'aggressive', 'cowardly', 'cruel', 'hateful', 'ferocious', 'honest', 'barbaric', 'dominant', 'nasty', 'insane', 'afraid', 'wicked', 'concerned', 'ill', 'horrible', 'insecure', 'hate', 'ruthless', 'irrational', 'powerful', 'cautious', 'unstable', 'perverse', 'disgusting', 'violent', 'uneasy', 'evil', 'hostile', 'delusional', 'nervous', 'troublesome', 'destructive', 'disrespectful', 'anxious', 'murderous', 'timid', 'fearful', 'dangerous', 'awful', 'mysterious', 'depressed', 'hopeless', 'difficult', 'mad', 'resistant', 'oppressive']


anger_homo = ['immoral', 'ridiculous', 'brutal', 'bad', 'punished', 'annoying', 'fierce', 'terrible', 'crazy', 'lonely', 'obnoxious', 'opposed', 'aggressive', 'cruel', 'hysterical', 'disappointed', 'guilty', 'hateful', 'illegal', 'ferocious', 'savage', 'BAD', 'worthless', 'barbaric', 'threatening', 'arrogant', 'nasty', 'insane', 'incompetent', 'disliked', 'offensive', 'offended', 'brazen', 'ill', 'angry', 'miserable', 'horrible', 'insecure', 'hate', 'ruthless', 'hurt', 'frustrated', 'poisonous', 'inept', 'powerful', 'insidious', 'defiant', 'upset', 'perverse', 'disgusting', 'violent', 'undesirable', 'evil', 'hostile', 'malicious', 'troublesome', 'horrific', 'suspicious', 'destructive', 'vicious', 'hot', 'harmful', 'murderous', 'despicable', 'awful', 'oppressive', 'unhappy', 'depressed', 'mad', 'selfish', 'controversial', 'intolerable']
disgust_homo= ['immoral', 'ridiculous', 'sick', 'bad', 'punished', 'appalling', 'fierce', 'pious', 'terrible', 'messy', 'lonely', 'obnoxious', 'cruel', 'ugly', 'disappointed', 'hateful', 'illegal', 'ferocious', 'BAD', 'worthless', 'barbaric', 'toxic', 'threatening', 'arrogant', 'nasty', 'pathetic', 'weird', 'discriminating', 'abnormal', 'alien', 'offensive', 'ill', 'angry', 'indecent', 'miserable', 'horrible', 'hate', 'ruthless', 'grotesque', 'unpopular', 'irrational', 'poisonous', 'inept', 'powerful', 'insidious', 'perverse', 'disgusting', 'fat', 'violent', 'ashamed', 'wretched', 'undesirable', 'evil', 'hostile', 'malicious', 'hypocritical', 'horrific', 'dirty', 'destructive', 'vicious', 'harmful', 'murderous', 'despicable', 'gross', 'ignorant', 'awful', 'shameless', 'oppressive', 'unpleasant', 'filthy', 'hideous', 'unhappy', 'mad', 'selfish']
fear_homo =['immoral', 'frightened', 'brutal', 'bad', 'punished', 'appalling', 'fierce', 'terrible', 'crazy', 'lonely', 'opposed', 'loyal', 'aggressive', 'cowardly', 'cruel', 'hysterical', 'hateful', 'isolated', 'illegal', 'ferocious', 'savage', 'BAD', 'barbaric', 'dominant', 'threatening', 'nasty', 'insane', 'afraid', 'wicked', 'alien', 'ill', 'horrible', 'insecure', 'hate', 'ruthless', 'hurt', 'irrational', 'poisonous', 'powerful', 'insidious', 'radioactive', 'perverse', 'disgusting', 'violent', 'undesirable', 'evil', 'hostile', 'malicious', 'nervous', 'troublesome', 'horrific', 'destructive', 'anxious', 'harmful', 'murderous', 'timid', 'fearful', 'dangerous', 'awful', 'mysterious', 'oppressive', 'hideous', 'depressed', 'hopeless', 'difficult', 'mad']

anger_liberal = ['ridiculous', 'rabid', 'brutal', 'bad', 'shit', 'annoying', 'fierce', 'terrible', 'crazy', 'dishonest', 'obnoxious', 'aggressive', 'cruel', 'hysterical', 'disappointed', 'guilty', 'hateful', 'ferocious', 'savage', 'honest', 'worthless', 'arrogant', 'nasty', 'insane', 'incompetent', 'disliked', 'offensive', 'offended', 'brazen', 'ill', 'angry', 'miserable', 'horrible', 'insecure', 'hate', 'ruthless', 'phony', 'frustrated', 'poisonous', 'inept', 'powerful', 'MAD', 'defiant', 'upset', 'screwed', 'unfair', 'disgusting', 'violent', 'evil', 'hostile', 'delusional', 'suspicious', 'destructive', 'vicious', 'hot', 'furious', 'murderous', 'despicable', 'awful', 'unhappy', 'mad', 'selfish', 'controversial', 'intolerable']
disgust_liberal = ['ridiculous', 'sick', 'rabid', 'bad', 'shit', 'fierce', 'terrible', 'dishonest', 'obnoxious', 'cruel', 'ugly', 'disappointed', 'hateful', 'ferocious', 'honest', 'worthless', 'toxic', 'arrogant', 'nasty', 'pathetic', 'weird', 'offensive', 'ill', 'angry', 'miserable', 'horrible', 'hate', 'ruthless', 'phony', 'irrational', 'unpopular', 'poisonous', 'inept', 'powerful', 'MAD', 'unfair', 'disgusting', 'fat', 'violent', 'evil', 'hostile', 'hypocritical', 'destructive', 'vicious', 'furious', 'murderous', 'despicable', 'ignorant', 'awful', 'shameless', 'unhappy', 'mad', 'selfish', 'greedy']
fear_liberal = ['serious', 'rabid', 'frightened', 'brutal', 'bad', 'fierce', 'terrible', 'crazy', 'loyal', 'aggressive', 'cowardly', 'cruel', 'hysterical', 'hateful', 'ferocious', 'savage', 'honest', 'dominant', 'nasty', 'insane', 'afraid', 'wicked', 'concerned', 'ill', 'horrible', 'insecure', 'hate', 'ruthless', 'irrational', 'poisonous', 'powerful', 'MAD', 'disgusting', 'violent', 'fragile', 'evil', 'hostile', 'delusional', 'nervous', 'destructive', 'anxious', 'timid', 'murderous', 'fearful', 'dangerous', 'awful', 'hopeless', 'difficult', 'mad', 'resistant']

#compute_diff(anger_rel, anger_homo, anger_liberal)
#compute_diff(disgust_rel, disgust_homo, disgust_liberal)
#compute_diff(fear_rel, fear_homo, fear_liberal)

anger_p = ['homeless', 'devastating', 'brutal', 'bad', 'shit', 'hurting', 'abandoned', 'fierce', 'terrible', 'crazy', 'lonely', 'cursed', 'cruel', 'ruined', 'armed', 'displaced', 'hysterical', 'disappointed', 'guilty', 'disgruntled', 'savage', 'worthless', 'unequal', 'deserted', 'threatening', 'nasty', 'painful', 'insane', 'incompetent', 'disliked', 'ill', 'angry', 'miserable', 'horrible', 'insecure', 'suicidal', 'neglected', 'hurt', 'frustrated', 'powerful', 'grim', 'unlucky', 'infamous', 'upset', 'screwed', 'unfair', 'violent', 'evil', 'hostile', 'delusional', 'broken', 'destroyed', 'hit', 'suspicious', 'agitated', 'chaotic', 'hot', 'furious', 'awful', 'bloody', 'unhappy', 'depressed', 'mad', 'selfish', 'socialist', 'controversial', 'powerless']
disgust_p = ['homeless', 'sick', 'devastating', 'bad', 'shit', 'fierce', 'terrible', 'messy', 'lonely', 'backwards', 'cruel', 'ruined', 'ugly', 'disappointed', 'disgruntled', 'worthless', 'toxic', 'unequal', 'deserted', 'threatening', 'nasty', 'painful', 'pathetic', 'weird', 'ill', 'angry', 'miserable', 'horrible', 'suicidal', 'neglected', 'unpopular', 'powerful', 'grim', 'unlucky', 'infamous', 'dismal', 'unfair', 'fat', 'violent', 'uneasy', 'ashamed', 'wretched', 'evil', 'hostile', 'dirty', 'suffering', 'furious', 'ignorant', 'awful', 'bloody', 'filthy', 'unhappy', 'mad', 'selfish', 'dire', 'socialist', 'greedy', 'powerless']
fear_p = ['homeless', 'devastating', 'frightened', 'brutal', 'bad', 'turbulent', 'endangered', 'forgotten', 'hurting', 'formidable', 'abandoned', 'fierce', 'terrible', 'crazy', 'lonely', 'cursed', 'cruel', 'ruined', 'armed', 'displaced', 'hysterical', 'isolated', 'savage', 'unequal', 'deserted', 'threatening', 'battered', 'nasty', 'painful', 'insane', 'afraid', 'concerned', 'ill', 'horrible', 'insecure', 'suicidal', 'hurt', 'powerful', 'grim', 'distressed', 'radioactive', 'unlucky', 'unstable', 'infamous', 'haunted', 'dismal', 'bankrupt', 'violent', 'uneasy', 'fragile', 'evil', 'hostile', 'delusional', 'broken', 'precarious', 'destroyed', 'broke', 'nervous', 'unsafe', 'helpless', 'unemployed', 'suffering', 'anxious', 'fearful', 'dangerous', 'awful', 'mysterious', 'bloody', 'depressed', 'hopeless', 'difficult', 'mad', 'doomed', 'resistant', 'dire', 'socialist', 'powerless']
sad_p = ['unlucky', 'devastating', 'nasty', 'ruined', 'terrible', 'barren', 'pathetic', 'upset', 'bleak', 'bad', 'black', 'distraught', 'helpless', 'mad', 'forgotten', 'dark', 'broke', 'haunted', 'hurting', 'hurt', 'ill', 'guilty', 'dismal', 'crazy', 'wretched', 'quiet', 'crippled', 'fragile', 'disappointed', 'evil', 'cursed', 'unpopular', 'disliked', 'bankrupt', 'unfair', 'exhausted', 'tough', 'socialist', 'unequal', 'neglected', 'depressed', 'displaced', 'broken', 'deserted', 'sick', 'unemployed', 'dire', 'doomed', 'painful', 'late', 'worthless', 'hopeless', 'grim', 'battered', 'awful', 'miserable', 'precarious', 'gloomy', 'negative', 'homeless', 'incompetent', 'ashamed', 'suicidal', 'bloody', 'concerned', 'insecure', 'badly', 'suffering', 'isolated', 'fearful', 'disconnected', 'destroyed', 'lonely', 'fat', 'unhappy', 'disgruntled', 'cruel', 'blue', 'worried', 'abandoned', 'powerless', 'lost']


anger_g = ['homeless', 'shaky', 'devastating', 'brutal', 'bad', 'wasteful', 'shit', 'annoying', 'terrible', 'crazy', 'lonely', 'cursed', 'cruel', 'ruined', 'disappointed', 'guilty', 'disgruntled', 'hateful', 'savage', 'honest', 'worthless', 'unequal', 'deserted', 'threatening', 'arrogant', 'nasty', 'painful', 'insane', 'incompetent', 'ill', 'angry', 'miserable', 'horrible', 'insecure', 'suicidal', 'neglected', 'paralyzed', 'hurt', 'frustrated', 'inept', 'powerful', 'failing', 'grim', 'MAD', 'noisy', 'unlucky', 'infamous', 'upset', 'screwed', 'unfair', 'violent', 'undesirable', 'evil', 'hostile', 'broken', 'destroyed', 'troublesome', 'suspicious', 'vicious', 'chaotic', 'hot', 'furious', 'awful', 'dreadful', 'bloody', 'unhappy', 'depressed', 'mad', 'selfish', 'socialist', 'controversial', 'uncertain', 'powerless']
disgust_g = ['homeless', 'sick', 'devastating', 'bad', 'wasteful', 'shit', 'bleeding', 'pious', 'terrible', 'messy', 'lonely', 'backwards', 'cruel', 'ruined', 'ugly', 'disappointed', 'disgruntled', 'hateful', 'honest', 'worthless', 'toxic', 'unequal', 'deserted', 'threatening', 'arrogant', 'nasty', 'painful', 'pathetic', 'weird', 'ill', 'angry', 'miserable', 'horrible', 'suicidal', 'neglected', 'unpopular', 'irrational', 'inept', 'powerful', 'grim', 'MAD', 'unlucky', 'infamous', 'dismal', 'unfair', 'fat', 'violent', 'ashamed', 'wretched', 'undesirable', 'evil', 'hostile', 'depressing', 'dirty', 'vicious', 'suffering', 'furious', 'ignorant', 'awful', 'dreadful', 'bloody', 'unpleasant', 'filthy', 'hideous', 'unhappy', 'mad', 'selfish', 'dire', 'socialist', 'greedy', 'uncertain', 'powerless']
fear_g = ['serious', 'homeless', 'shaky', 'devastating', 'frightened', 'brutal', 'bad', 'endangered', 'formidable', 'bleeding', 'terrible', 'crazy', 'lonely', 'cursed', 'cowardly', 'cruel', 'ruined', 'isolated', 'hateful', 'savage', 'honest', 'unequal', 'deserted', 'threatening', 'nasty', 'painful', 'insane', 'afraid', 'concerned', 'risky', 'ill', 'horrible', 'insecure', 'suicidal', 'paralyzed', 'hurt', 'irrational', 'powerful', 'failing', 'grim', 'MAD', 'distressed', 'unlucky', 'unstable', 'infamous', 'dismal', 'bankrupt', 'violent', 'fragile', 'undesirable', 'evil', 'hostile', 'broken', 'destroyed', 'broke', 'nervous', 'troublesome', 'enigmatic', 'unsafe', 'helpless', 'unemployed', 'suffering', 'anxious', 'timid', 'fearful', 'dangerous', 'awful', 'dreadful', 'mysterious', 'bloody', 'hideous', 'depressed', 'hopeless', 'difficult', 'mad', 'doomed', 'resistant', 'dire', 'socialist', 'uncertain', 'powerless']
sad_g = ['unlucky', 'devastating', 'nasty', 'MAD', 'ruined', 'timid', 'terrible', 'barren', 'pathetic', 'upset', 'bleak', 'bad', 'black', 'helpless', 'mad', 'dreadful', 'dark', 'broke', 'depressing', 'failing', 'hurt', 'ill', 'guilty', 'dismal', 'crazy', 'serious', 'wretched', 'quiet', 'crippled', 'pious', 'fragile', 'disappointed', 'evil', 'cursed', 'unpopular', 'crumbling', 'inefficient', 'wasteful', 'bankrupt', 'unfair', 'tough', 'socialist', 'unequal', 'paralyzed', 'hateful', 'neglected', 'depressed', 'broken', 'deserted', 'sick', 'unemployed', 'dire', 'bleeding', 'unpleasant', 'doomed', 'painful', 'late', 'worthless', 'undesirable', 'hopeless', 'grim', 'awful', 'miserable', 'sluggish', 'gloomy', 'negative', 'stagnant', 'homeless', 'honest', 'incompetent', 'feeble', 'hideous', 'ashamed', 'suicidal', 'bloody', 'concerned', 'insecure', 'badly', 'suffering', 'isolated', 'fearful', 'disconnected', 'destroyed', 'lonely', 'disappointing', 'fat', 'unhappy', 'disgruntled', 'cruel', 'blue', 'worried', 'powerless', 'dull']

#cd(anger_p, anger_g)
#cd(disgust_p, disgust_g)
#cd(fear_p, fear_g)
#cd(sad_p, sad_g)

# ------------------------BART

anger_l = ['selfish', 'unpopular', 'contempt', 'vicious', 'ruthless', 'toxic', 'nasty', 'weird', 'inept', 'sick', 'ugly', 'interested', 'honest', 'unhappy', 'powerful', 'furious', 'cruel', 'hostile', 'ridiculous', 'distrust', 'miserable', 'angry', 'arrogant', 'awful', 'hypocritical', 'dishonest', 'hateful', 'bad', 'evil', 'terrible', 'pathetic', 'offensive', 'ignorant', 'irrational', 'violent', 'greedy', 'horrible', 'hate', 'damn', 'mad', 'politic', 'despicable']
disgust_l =['crazy', 'contempt', 'ruthless', 'difficult', 'nasty', 'dangerous', 'dominant', 'honest', 'anxious', 'resistant', 'powerful', 'frightened', 'cruel', 'opposed', 'hostile', 'afraid', 'cautious', 'concerned', 'distrust', 'awful', 'hateful', 'insecure', 'bad', 'evil', 'terrible', 'aggressive', 'irrational', 'insane', 'violent', 'cowardly', 'fearful', 'horrible', 'reluctant', 'hate', 'timid', 'nervous', 'loyal', 'mad']

#cd(anger_liberal, anger_l)
#cd(disgust_liberal, disgust_l)