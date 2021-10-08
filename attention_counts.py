import os
import nltk, re, string
import transformers
import datasets
import pandas as pd
import numpy as np
from transformers import BertTokenizer, BertModel
from datasets import load_dataset, DatasetDict, dataset_dict
from nltk.tokenize import word_tokenize
from nltk.text import ConcordanceIndex

# TODO think about setting up python 3.7 env as rochelle was using

# need the corpus to get the concordances

def nltk_preprocess(data, tokenize=True):
    ''' 
    input list of strings with each string being one article
    '''
    # took inspiration from: https://www.kaggle.com/rtatman/tutorial-getting-n-grams
    tmp = ' '.join(data)
    punctuationNoPeriod = "[" + re.sub("\.","",string.punctuation) + "]"
    tmp = re.sub(punctuationNoPeriod, "", tmp).lower()
    if tokenize:
        tmp = word_tokenize(tmp)
#    return tmp.split()
    return tmp


def find_concordance_custom(ci, word, width=80):
        """
        Find all concordance lines given the query word.
        Provided with a list of words, these will be found as a phrase.
        """
        # rewrite find_concordance. http://www.nltk.org/_modules/nltk/text.html#ConcordanceIndex
        if isinstance(word, list):
            phrase = word
        else:
            phrase = [word]

        half_width = (width - len(' '.join(phrase)) - 2) // 2
        context = width // 4  # approx number of words of context

        # Find the instances of the word to create the ConcordanceLine
        concordance_list = []
        offsets = ci.offsets(phrase[0])
        for i, word in enumerate(phrase[1:]):
            word_offsets = {offset - i - 1 for offset in ci.offsets(word)}
            offsets = sorted(word_offsets.intersection(offsets))
        if offsets:
            for i in offsets:
                query_word = " ".join(ci._tokens[i : i + len(phrase)])
                # Find the context of query word.
                left_context = ci._tokens[max(0, i - context) : i]
                right_context = ci._tokens[i + len(phrase) : i + context]
                # Create the pretty lines with the query_word in the middle.
                left_print = " ".join(left_context)[-half_width:]
                right_print = " ".join(right_context)[:half_width]
                # The WYSIWYG line of the concordance.
                line_print = " ".join([left_print, query_word, right_print])
                # Create the ConcordanceLine
#                 concordance_line = ConcordanceLine(
#                     left_context,
#                     query_word,
#                     right_context,
#                     i,
#                     left_print,
#                     right_print,
#                     line_print,
#                 )
#                 concordance_list.append(concordance_line)
                concordance_list.append(line_print)
        return concordance_list

def train_val_split(dat, validation_percentage=0.1):
    ''' 
    adapted from https://discuss.huggingface.co/t/how-to-split-main-dataset-into-train-dev-test-as-datasetdict/1090/2
    simple function to split existing huggingface Dataset object. 
    returns DatasetDict containing training and validation dataset
    default split ratio 90-10
    note that validation set is still named test as I am using datasets train_test_split method
    '''
    train_valid = dat.train_test_split(test_size=0.1)
    # gather everyone to have a single DatasetDict
    #train_valid_dict = DatasetDict({
    #    'train': train_valid['train'],
    #    'valid': train_valid['test']})
    return train_valid

def get_concordances(cat_of_interest, newspaper):
    
    # TODO make dataset path an input arg?

    allthenews = load_dataset('csv', script_version='master', data_files=['../bias/data/external/archive/articles1.csv', '../bias/data/external/archive/articles2.csv', '../bias/data/external/archive/articles3.csv'], 
                          column_names = ['Unnamed', 'id', 'title', 'publication', 'author', 'date', 'year', 'month', 'url', 'content'])
    
    # maybe do this for all newspapers simultaneously in the future, for now stick to one
    #allthenews_dict = {newspaper : allthenews.filter(lambda example: example['publication']==newspaper).remove_columns(['Unnamed', 'title', 'publication', 'author', 'year', 'month', 'url', 'date']) for newspaper in newspapers}

    # filter for 'newspaper' keep all samples (that were before in train)
    try:
        news = allthenews.filter(lambda example: example['publication']==newspaper).remove_columns(['Unnamed', 'title', 'publication', 'author', 'year', 'month', 'url', 'date'])['train']['content']
    except:
        raise Exception('You entered an invalid newspaper. Enter one of Breitbart, Fox News, Guardian or Reuters.')
    
    # tokenize using nltk
    tokenized_news = nltk_preprocess(news)
1
    # allthebreitbart = allthenews.filter(lambda example: example['publication']=='Breitbart').remove_columns(['Unnamed', 'title', 'publication', 'author', 'year', 'month', 'url', 'date'])
    # allthefox = allthenews.filter(lambda example: example['publication']=='Fox News').remove_columns(['Unnamed', 'title', 'publication', 'author', 'year', 'month', 'url', 'date'])
    # allthereuters = allthenews.filter(lambda example: example['publication']=='Reuters').remove_columns(['Unnamed', 'title', 'publication', 'author', 'year', 'month', 'url', 'date'])
    # alltheguardian = allthenews.filter(lambda example: example['publication']=='Guardian').remove_columns(['Unnamed', 'title', 'publication', 'author', 'year', 'month', 'url', 'date'])
    # allthenews_dict = {'breitbart': allthebreitbart, 'fox': allthefox, 'reuters': allthereuters, 'guardian': alltheguardian}#, 'newyorker': allthenewyorker}
    # allthenews_trainval = {newspaper : train_val_split(allthenews_dict[newspaper]['train']) for newspaper, dat in allthenews_dict.items()}
    # tokenized_news = {newspaper: nltk_preprocess(allthenews_trainval[newspaper]['train']['content']) for newspaper in newspapers}
    #tokenized_news = {newspaper: nltk_preprocess(allthenews_trainval[newspaper]['train']['content']) for newspaper in ['guardian']}

    ci = ConcordanceIndex(tokenized_news)
    concordances = find_concordance_custom(ci, cat_of_interest)
    return concordances


def group_tokens_to_ids(tokens, tokenizer):
    ''' return dict {token : id} if tokens is a list of tokens 
    or a dict {category : {token : id} } if tokens is a dict of type {category : token}
    tokens that are not recognised and are mapped to the id 100 for the unknown token are left out'''

    if type(tokens) == list:
        ids = {token: idx for token, idx in zip(tokens, tokenizer.convert_tokens_to_ids(tokens))}
    if type(tokens) == dict:
        ids = {cat: {token: idx for token, idx in zip(tokens[cat], tokenizer.convert_tokens_to_ids(tokens[cat])) if idx != 100} for cat in tokens.keys()}
    return ids


def attention_weighted_counts(concordances, reference_word, bert_model_path, layer_id, attention_head_id):
    '''
    Given a category of interest (reference_word) and a list of strings that each contain reference word count the emotion words
    Counts are weighted by the attention when passing concordances through an LM
    use attention head number 'attention_head_id' in layer 'layer_id'.
    
    Example Usage:
    attention_weighted_counts(concordances_fox_girl, 'girl', 'bert-base-uncased', emotionwords_dict, 11, 0)
    '''

    # TODO change these to AutoTokenizer and AutoModelForMaskedLM later and the input arg from bert_model_path to model_path for clarity
    print('--Calculating counts weighted by attention--')
    tokenizer = BertTokenizer.from_pretrained(bert_model_path)#, do_lower_case=True)
    model = BertModel.from_pretrained(bert_model_path, output_attentions=True)

    print('--tokenizing concordances--')
    tokenized_concordances = tokenizer(concordances, padding=True, return_tensors='pt')
    token_type_ids = tokenized_concordances['token_type_ids']
    input_ids = tokenized_concordances['input_ids']
    
    print('--getting attention--')
    attentions = model(input_ids, token_type_ids=token_type_ids)[-1]

    #load emotion words
    emotionwords = pd.read_excel('../data/external/NRC-Emotion-Lexicon-v0.92-In105Languages-Nov2017Translations.xlsx', usecols="A,DB:DK")
    emotions = list(emotionwords.columns[1:])
    emotionwords_dict = {emotion: list(emotionwords['English (en)'].loc[emotionwords[emotion]==1]) for emotion in emotions}
    # cast series to list. change back if this breaks things
    del emotionwords
    
    #cts = attention_weighted_counts(tokenized_concordances['input_ids'], tokenizer, attentions, reference_word, emotion_tokens, layer_id, attention_head_id)
    #print(attentions.shape)
    
    # get id of identity_word that serves as reference point
    ref_wd_id = tokenizer.convert_tokens_to_ids(reference_word)
    
    # number of times identity word occurrs. should be length of text_snippets
    N = len(concordances)

    # get list of lists with tuples (word_idx, attention from reference word to this word)
    print('--collecting counts--')
    w_cts = [[(idx.item(), attention.item()/N) for idx, attention in zip(input_ids[sentence_id], attentions[layer_id][sentence_id, attention_head_id, input_ids[sentence_id].tolist().index(ref_wd_id),:])] for sentence_id in range(input_ids.shape[0])]
        
    # flatten list
    w_cts = [item for sublist in w_cts for item in sublist]
    
    # emotion tokens to ids
    print('--mapping emotion word tokens to ids--')
    emotion_ids = group_tokens_to_ids(emotionwords_dict, tokenizer)
        
    print('--summing over emotion categories--')
    sum_w_cts = {emotion: sum([sum([ct[1] for ct in w_cts if ct[0] == emotion_term_id]) for emotion_term, emotion_term_id in emotion_ids[emotion].items()]) for emotion in emotions}
    
    print('--finished weighted counts of emotion words--')

    return sum_w_cts


def show_counts(newspapers, gps_of_interest, model_path, layer_id, attention_head_id):
    '''
    newspapers, gps_of_interest: str or list of strings with newspapers and groups, ie words designating different groups that we are interested in
    
    Example usage: show_counts('Guardian', 'girl', 'bert-base-uncased', 11, 0)
    '''
    # test that this works first
    if type(newspapers) == str and type(gps_of_interest) == str and type(model_path) == str:
        concordances = get_concordances(gps_of_interest, newspapers)
        cts = attention_weighted_counts(concordances, gps_of_interest, model_path, layer_id, attention_head_id)
        print(newspapers, gps_of_interest)
        print("{:<8} {:<15}".format('Emotion','Score'))
        for k, v in cts.items():
            print("{:<8} {:<15}".format(k, v))



# TODO define main

# TODO is this really the best place for this: 
# newspapers = ['breitbart', 'fox', 'reuters', 'guardian']
newspapers = ['Breitbart', 'Fox News', 'Reuters', 'Guardian']

# gender terms
female_terms = ["girls", "women", "females", "girlfriends", "stepmothers", "ladies", "sisters", "mothers", "grandmothers" "wives", "brides", "schoolgirls", "mommies"]
male_terms = ["men", "males", "boys" "boyfriends", "stepfathers", "gentlemen" "brothers", "fathers", "grandfathers", "husbands", "grooms",  "schoolboys",  "daddies"]

# deciding to add some synonyms, singulars, plurals etc
female_terms = ["she", "her", "girl", "girls", "woman", "women", "female", "females", "girlfriend", "girlfriends", "stepmothers", "lady", "ladies", "sister", "sisters", "mother", "mothers", "grandmothers", "wife", "wives", "bride", "brides", "schoolgirls", "mom", "mum", "moms", "mums", "mummies", "mommies", "miss", "mrs", "ms", "lady", "mistress"]
male_terms = ["he", "his", "him", "boy", "boys", "man", "men", "male", "males", "boyfriend", "boyfriends", "stepfathers", "gentleman", "gentlemen", "brother", "brothers", "father", "fathers", "grandfathers", "husband", "husbands", "groom", "grooms",  "schoolboys", "dad", "dads", "daddy", "daddies", "mr", "sir", "lord"]
neutral_terms = ["they", "their", "them", "child", "person", "people", "parent", "parents", "partner", "partners", "spouse", "sibling", "siblings"]
female_terms_short = ["she", "woman", "girl"]#her", "girl", "girls", "woman", "women", "female"]
male_terms_short = ["he", "man", "boy"]#his", "him", "boy", "boys", "man", "men", "male"]

race_terms = list(map(lambda x: x.lower(), ["Asians", "Americans", "Europeans", "Jews", "Indians", "Russians", "Africans", "Black people", "Mexicans", "Whites", "Blacks", "White people", "Germans", "blondes", "blonde girls", "Arabs", "White Americans", "Black Americans", "Hispanics", "Native Americans", "Black men", "White men", "Asian women", "Asian men", "Black women", "the Dutch", "Irish people", "Irish men", "White women", "Indian men", "Indian women", "Chinese men", "Chinese women", "Japanese women", "Japanese men", "Indian parents", "Asian parents", "White parents", "Black parents", "Black fathers", "Latinas", "Latinos", "Latin people", "Brazilian women","Asian kids", "Black kids", "White kids", "African Americans", "Nigerians", "Ethiopians", "Ukrainians", "Sudanese people", "Afghans", "Iraqis", "Hispanic men", "Hispanic women", "Italians", "Italian men", "Italian women", "Somalis", "Iranian people", "Iranians", "Australians", "Australian men", "Australian women", "Aussies", "Ghanaians", "Swedes", "Finns", "Venezuelans", "Moroccans", "Syrians", "Pakistanis", "British people", "French people", "Greeks", "Indonesians", "Vietnamese people", "Romanians", "Ecuadorians", "Norwegians", "Nepalis" , "Scots", "Bengalis", "Polish people", "Taiwanese people", "Albanians", "Colombians", "Egyptians", "Koreans", "Persian people", "Portuguese men", "Portuguese women", "Turkish people", "Austrians", "South Africans", "Dutch people", "Chileans", "Lebanese people"]))
race_terms_short = list(map(lambda x: x.lower(), ["Asian", "American", "European", "Jewish", "Indian", "African", "Black", "Mexican", "White", "Arab"]))
social_gps_paper = list(map(lambda x: x.lower(), ['christian', 'police', 'conservative', 'celebrities', 'gay', 'academics', 'Iraq', 'asian', 'black', 'ladies', 'teenager']))