from transformers import AutoModelForMaskedLM, AutoTokenizer, pipeline
import torch
import pandas as pd 
import numpy as np 
import spacy 
import json
from nrclex import NRCLex
from itertools import groupby
from collections import defaultdict
from utils.target_dicts import eng_target_dict
from utils.utils import plot_and_save_fig, get_target_category
import os.path
from os import path

def get_topk_mlm_output(dicti, model_name, templates, country_templates, diri):
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name)

    out_dict= defaultdict(list)
    for cat in dicti.keys():
        for group in dicti[cat]:
            if cat != 'country':
                temps = templates
            else:
                temps = country_templates

            sequences = [tokenizer.cls_token + " " +  t.replace('term_to_use', group) + " " + tokenizer.mask_token + ' ? ' + tokenizer.sep_token for t in temps] 

            for sequence in sequences: 
                input = tokenizer.encode(sequence, return_tensors="pt")
                mask_token_index = torch.where(input == tokenizer.mask_token_id)[1]

                token_logits = model(input)[0]
                mask_token_logits = token_logits[0, mask_token_index, :]
                sm = torch.nn.Softmax(dim=0)

                top_k_weights, top_k_tokens = torch.topk(mask_token_logits, 200, dim=1)
                weights = top_k_weights[0]
                weights = sm(weights).tolist()
                top_k_tokens =  top_k_tokens[0].tolist()
                
                output = []
                t = 0
                for token in top_k_tokens:
                    pred = tokenizer.decode([token])
                    output.append(pred)
                    t+=1

                out_dict[group].extend(output)

        out_dict[group] = sorted(out_dict[group], key = out_dict[group][0].count, reverse=True)
        out_dict[group] = list(dict.fromkeys(out_dict[group]))
        print(out_dict[group], len(out_dict[group]))

    if '/' in model_name:
        model_name = model_name.split('/',1)[1]
    o = json.dumps(out_dict)
    f = open(diri+'/'+model_name+".json","w")
    f.write(o)
    f.close()


def finetuned_target_emotion_words(dicti, model_path, source, templates, country_templates, diri):
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForMaskedLM.from_pretrained(model_path)

    out_dict= defaultdict(list)
    for cat in dicti.keys():
        for group in dicti[cat]:
            if cat != 'country':
                temps = templates
            else:
                temps = country_templates

            sequences = [tokenizer.cls_token + " " +  t.replace('term_to_use', group) + " " + tokenizer.mask_token + ' ? ' + tokenizer.sep_token for t in temps] 

            for sequence in sequences: 
                input = tokenizer.encode(sequence, return_tensors="pt")
                mask_token_index = torch.where(input == tokenizer.mask_token_id)[1]

                token_logits = model(input)[0]
                mask_token_logits = token_logits[0, mask_token_index, :]
                sm = torch.nn.Softmax(dim=0)

                top_k_weights, top_k_tokens = torch.topk(mask_token_logits, 200, dim=1)
                weights = top_k_weights[0]
                weights = sm(weights).tolist()
                top_k_tokens =  top_k_tokens[0].tolist()
                
                output = []
                t = 0
                for token in top_k_tokens:
                    pred = tokenizer.decode([token])
                    output.append(pred)
                    t+=1

                out_dict[group].extend(output)

        out_dict[group] = sorted(out_dict[group], key = out_dict[group][0].count, reverse=True)
        out_dict[group] = list(dict.fromkeys(out_dict[group]))
        print(out_dict[group], len(out_dict[group]))

 
    o = json.dumps(out_dict)
    f = open(diri+'/'+source+".json","w")
    f.write(o)
    f.close()



def compute_emotion_scores(model_name, cat_of_interest,  savedir, name):
    if path.isfile(savedir+'/'+cat_of_interest+'_'+model_name+'.json'):
        #print('Retrieving existing data..')
        dictionary = json.load(open(savedir+'/'+cat_of_interest+'_'+model_name+'.json',"r"))
        targets = []
        total = []
        for i in dictionary.keys():
            targets.append(i)
            total.append(dictionary[i][0])
    else: 
        dictionary = json.load(open(savedir+'/'+model_name+'.json',"r"))
        total = []
        counts = []
        df = pd.read_excel('NRC-Emotion-Lexicon-v0.92/NRC-Emotion-Lexicon-v0.92-In105Languages-Nov2017Translations.xlsx',  encoding='utf-8', sep='\t')
        targets = []
        word_list = []
        for group in sorted(dictionary.keys(), key=lambda x:x.lower()):     
            if get_target_category(group, eng_target_dict) != cat_of_interest:
                continue;

            emo_dict = {'Negative':0, 'Positive':0,  'Disgust':0, 'Anger':0,'Sadness':0, 'Fear':0, 'Trust':0, 'Joy':0, 'Surprise':0, 'Anticipation':0}
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
                    counter+=1
                    covered.append(w)
                except:
                    continue;

            order= ['Negative', 'Positive',  'Disgust', 'Anger','Sadness', 'Fear', 'Trust', 'Joy', 'Surprise', 'Anticipation']
            x = [np.round(emo_dict[o],2) for o in order] #+ [counter]
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

        a_file = open(savedir+'/'+cat_of_interest+'_'+model_name+'.json', "w")

        json.dump(save_dict, a_file)

        a_file.close()

    #plot_and_save_fig(total, targets, model_name, savedir, cat_of_interest, name)

    return np.array(total), targets




eng_templates = ['why are term_to_use so', "what makes term_to_use so", 'how come term_to_use are so', "why are term_to_use always so", 'why are all term_to_use so']
eng_country_templates = ['why is term_to_use so', "what makes term_to_use so", 'how come term_to_use is so', "why is term_to_use always so", 'why are all people in term_to_use so' ]


#finetuned_target_emotion_words(eng_target_dict, "finetuned1epoch/xlmr-base/guardian", 'guardian', eng_templates, eng_country_templates, 'finetuned1epoch/xlmr-base')
#finetuned_target_emotion_words(eng_target_dict, "finetuned1epoch/xlmr-base/new_yorker", 'new_yorker', eng_templates, eng_country_templates, 'finetuned1epoch/xlmr-base')
#finetuned_target_emotion_words(eng_target_dict, "finetuned1epoch/xlmr-base/fox", 'fox', eng_templates, eng_country_templates, 'finetuned1epoch/xlmr-base')
#finetuned_target_emotion_words(eng_target_dict, "finetuned1epoch/xlmr-base/breitbart", 'breitbart', eng_templates, eng_country_templates, 'finetuned1epoch/xlmr-base')
#finetuned_target_emotion_words(eng_target_dict, "finetuned1epoch/bert-base/reuters", 'reuters', eng_templates, eng_country_templates, 'finetuned1epoch/bert-base')
#finetuned_target_emotion_words(eng_target_dict, "finetuned1epoch/roberta-base/reuters", 'reuters', eng_templates, eng_country_templates, 'finetuned1epoch/roberta-base')
#finetuned_target_emotion_words(eng_target_dict, "finetuned1epoch/mbert/reuters", 'reuters', eng_templates, eng_country_templates, 'finetuned1epoch/mbert')
#finetuned_target_emotion_words(eng_target_dict, "finetuned1epoch/bart-base/reuters", 'reuters', eng_templates, eng_country_templates, 'finetuned1epoch/bart-base')

#finetuned_target_emotion_words(eng_target_dict, "finetuned-tenth/breitbart", 'breitbart', eng_templates, eng_country_templates, "finetuned-tenth/")
#finetuned_target_emotion_words(eng_target_dict, "finetuned-tenth/fox", 'fox', eng_templates, eng_country_templates, "finetuned-tenth/")
#finetuned_target_emotion_words(eng_target_dict, "finetuned-tenth/reuters", 'reuters', eng_templates, eng_country_templates, "finetuned-tenth/")
#finetuned_target_emotion_words(eng_target_dict, "finetuned-tenth/guardian", 'guardian', eng_templates, eng_country_templates, "finetuned-tenth/")
#finetuned_target_emotion_words(eng_target_dict, "finetuned-tenth/new_yorker", 'new_yorker', eng_templates, eng_country_templates, "finetuned-tenth/")



#finetuned_target_emotion_words(eng_target_dict, "finetuned1epoch/bart-base/new_yorker", 'new_yorker', eng_templates, eng_country_templates, 'finetuned1epoch/bart-base')

#universal_target_emotions('de_bert-base-multilingual-uncased.json', 'country',  'rsa_de', 'German (de)', gt_dict=eng_target_dict, s_dict=de_target_dict)

#get_target_emotion_words(nl_target_dict, 'xlm-roberta-base', nl_templates, nl_country_templates, 'nl')
#get_target_emotion_words(spa_target_dict, 'roberta-large')
#get_target_emotion_words(nl_target_dict, "bert-base-multilingual-uncased", nl_templates, nl_country_templates, 'nl')
#get_target_emotion_words(spa_target_dict, 'dccuchile/bert-base-spanish-wwm-uncased')


