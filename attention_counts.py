import os, sys
import re, string
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset
from typing import Dict

# TODO think about setting up python 3.7 env as rochelle was using

def group_tokens_to_ids(tokens, tokenizer):
    ''' return dict {token : id} if tokens is a list of tokens 
    or a dict {category : {token : id} } if tokens is a dict of type {category : token}
    tokens that are not recognised and are mapped to the id 100 for the unknown token are left out'''

    if type(tokens) == list:
        ids = {token: idx for token, idx in zip(tokens, tokenizer.convert_tokens_to_ids(tokens))}
    if type(tokens) == dict:
        ids = {cat: {token: idx for token, idx in zip(tokens[cat], tokenizer.convert_tokens_to_ids(tokens[cat])) if idx != 100} for cat in tokens.keys()}
    return ids

def group_texts(examples, block_size=128):
    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
    total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result

def attention_weighted_counts(args: Dict):
    '''
    Given a category of interest (reference_word) and a list of strings that each contain reference word count the emotion words
    Counts are weighted by the attention when passing concordances through an LM
    use attention head number 'attention_head_id' in layer 'layer_id'.
    
    Example Usage:
    attention_weighted_counts(concordances_fox_girl, 'girl', 'bert-base-uncased', emotionwords_dict, 11, 0)
    '''

    newspaper = str(args['--newspaper'])
    reference_word = str(args['--reference_word'])
    model_path = str(args['--model_path'])
    layer_id = int(args['--layer_id'])
    attention_head_id = int(args['--attention_head_id'])

    #load emotion words
    emotionwords = pd.read_excel('../data/external/NRC-Emotion-Lexicon-v0.92-In105Languages-Nov2017Translations.xlsx', usecols="A,DB:DK")
    emotions = list(emotionwords.columns[1:])
    emotionwords_dict = {emotion: list(emotionwords['English (en)'].loc[emotionwords[emotion]==1]) for emotion in emotions}
    print('--loaded emotion words--')

    tokenizer = AutoTokenizer.from_pretrained(model_path)#, do_lower_case=True)
    model = AutoModel.from_pretrained(model_path, output_attentions=True)
    print('--defined model--')

    news = load_dataset('csv', script_version='master', data_files=['../data/external/archive/articles1.csv', '../data/external/archive/articles2.csv', '../data/external/archive/articles3.csv'],
                          column_names = ['Unnamed', 'id', 'title', 'publication', 'author', 'date', 'year', 'month', 'url', 'content']).remove_columns(['Unnamed', 'title', 'author', 'year', 'month', 'url', 'date'])
    print('--loaded dataset--')

    # filter for publication of interest
    news_newspaper = news.filter(lambda example: example['publication']==newspaper).remove_columns(['publication'])
    
    # find articles that contain the word
    articles_w_ref_wd = news_newspaper.filter(lambda example: reference_word in example['content'])

    # tokenize
    def tokenize_function(examples):
        return tokenizer(examples["content"])
    
    articles_tokenized = articles_w_ref_wd.map(tokenize_function, batched=True, num_proc=4, remove_columns=['id', 'content'])

    # split data into processable chunks
    snippets = articles_tokenized.map(
            group_texts,
            batched=True,
            batch_size=1000,
            num_proc=4,
        )

    # find idx of reference word
    idx = tokenizer.convert_tokens_to_ids(reference_word)

    # discard snippets that don't have the reference word
    snippets_wd = snippets['train'].filter(lambda example: idx in example['input_ids'])

    # cast to tensor
    input_ids = torch.tensor(snippets_wd['input_ids'])
    token_type_ids = torch.tensor(snippets_wd['token_type_ids'])
    print('--prepared concordances--')

    # pass the chunks to the model to get attention
    attentions = model(input_ids, token_type_ids=token_type_ids)[-1]
    print('--calculated attentions--')
    
    # number of times identity word occurrs. should be length of text_snippets
    N = len(snippets_wd)

    # get list of lists with tuples (word_idx, attention from reference word to this word)
    print('--collecting counts--')
    w_cts = [[(idx.item(), attention.item()/N) for idx, attention in zip(input_ids[sentence_id], attentions[layer_id][sentence_id, attention_head_id, input_ids[sentence_id].tolist().index(idx),:])] for sentence_id in range(input_ids.shape[0])]
        
    # flatten list
    w_cts = [item for sublist in w_cts for item in sublist]
    
    # emotion tokens to ids
    print('--mapping emotion word tokens to ids--')
    emotion_ids = group_tokens_to_ids(emotionwords_dict, tokenizer)
        
    print('--summing over emotion categories--')
    sum_w_cts = {emotion: sum([sum([ct[1] for ct in w_cts if ct[0] == emotion_term_id]) for emotion_term_id in emotion_ids[emotion].values()]) for emotion in emotions}
    
    print('--finished weighted counts of emotion words--')

    return sum_w_cts


def show_counts(newspaper, reference_word, model_path, layer_id, attention_head_id):
    '''
    newspapers, gps_of_interest: str or list of strings with newspapers and groups, ie words designating different groups that we are interested in
    
    Example usage: show_counts('Guardian', 'girl', 'bert-base-uncased', 11, 0)
    '''
    # test that this works first
    if type(newspapers) == str and type(reference_word) == str and type(model_path) == str:
        cts = attention_weighted_counts(newspaper, reference_word, model_path, layer_id, attention_head_id)
        print(newspapers, reference_word)
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