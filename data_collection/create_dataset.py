'''
	This file is used to create a dataset containing stereotypical autocomplete suggestions from three search engines:
	Google, Yahoo and DuckDuckGo. The publicly available API's are used for the resepective search engines to collect the data.
'''

# Import packages
import sys
import json
import logging
import requests
import numpy as np
import pandas as pd
from fake_useragent import UserAgent

# Import files
sys.path.append("..")
from utils.target_dicts import eng_target_dict, templates, country_templates


def data_collection(target_dict, templates, country_templates, func, savefile='test.csv'):
	'''
	Code for collecting and writing autocomplete suggestions.
	'''
	inputs = []
	completions = []
	target_group = []
	categories = []

	for category in target_dict.keys():
		if category != 'country':
			temps = templates
		else:
			temps = country_templates
		for group in target_dict[category]:
			for temp in templates: 
				query = temp.replace('term_to_use', group)
				keywords = query.replace(" ", "+")
				i, c, t, cat = func(keywords, query, temp, group, category)
				inputs += i
				completions += c
				target_group += t
				categories += cat

	df = pd.DataFrame(data={'input': inputs, 'target_category': categories, 'target_group': target_group, 
									'completion': completions, 'search_engine': [func.__name__]*len(inputs)})
	df.to_csv(savefile, index=False,  encoding='utf-8', sep='\t')

	logging.info('Num of completions: ', len(completions ))



def google(keywords, query, temp, group, cat):
	'''
	Code for querying autocomplete suggestion using the Google API.
	'''
	url = "http://suggestqueries.google.com/complete/search?output=chrome&q=" + keywords +"&hl=en"
	ua = UserAgent()
	headers = {"user-agent": ua.chrome}
	response = requests.get(url, headers=headers, verify=False)
	suggestions = json.loads(response.text)

	if suggestions[1] == []:
		logging.info(query, " NONE")
	else:
		logging.info(query, suggestions[1])
	
	inputs, completions, target_group, categories = [], [], [], []
	for pred in suggestions[1]:
		word_list, q = dissect_temp(temp)
		if check_query(word_list, pred, group):
			inputs.append(query.strip()+'\n')
			autocomplete = pred.split(q)
			completions.append(pred.split(q, 1)[len(autocomplete)-1]+'\n')
			target_group.append(group+'\n')
			categories.append(cat)

	return inputs, completions, target_group, categories

def yahoo(keywords, query, temp, group, cat):
	'''
	Code for querying autocomplete suggestion using Yahoo's API.
	'''
	url = 'http://sugg.search.yahoo.net/sg/?output=chromep&nresults=10&command=' + keywords +"&hl=en"
	ua = UserAgent()
	headers = {"user-agent": ua.chrome}
	response = requests.get(url, headers=headers, verify=False)
	suggestions = json.loads(response.text)

	if suggestions['gossip']['results'] == []:
		logging.info(query, " NONE")
	else:
		logging.info(query, suggestions['gossip']['results'][0]['key'])
		
	inputs, completions, target_group, categories = [], [], [], []
	for i in suggestions['gossip']['results']:
		pred = i['key']
		word_list, q = dissect_temp(temp)
		if check_query(word_list, pred, group):
			inputs.append(query.strip()+'\n')
			autocomplete = pred.split(q)
			completions.append(pred.split(q, 1)[len(autocomplete)-1]+'\n')
			target_group.append(group+'\n')
			categories.append(cat)

	return inputs, completions, target_group, categories

def duckduckgo(keywords, query, temp, group, cat):
	'''
	Code for querying autocomplete suggestion using the DuckDuckGo API.
	'''
	url = 'https://duckduckgo.com/ac/?q=' + keywords +"&hl=en"
	ua = UserAgent()
	headers = {"user-agent": ua.chrome}
	response = requests.get(url, headers=headers, verify=False)
	suggestions = json.loads(response.text)
	if suggestions == []:
		logging.info(query, " NONE")
	else:
		logging.info(query, suggestions)
		
	inputs, completions, target_group, categories = [], [], [], []
	for i in suggestions:
		pred = i['phrase']
		word_list, q = dissect_temp(temp)
		if check_query(word_list, pred, group):
			inputs.append(query.strip()+'\n')
			autocomplete = pred.split(q)
			completions.append(pred.split(q, 1)[len(autocomplete)-1]+'\n')
			target_group.append(group+'\n')
			categories.append(cat)

	return inputs, completions, target_group, categories


def merge_data(engine1, engine2, engine3, savefile):
	'''
	Merge data from different search engines into one file.
	'''
	df1 = pd.read_csv(engine1+'_data.csv', encoding='utf-8', sep='\t')
	df2 = pd.read_csv(engine2+'_data.csv', encoding='utf-8', sep='\t')
	df3 = pd.read_csv(engine3+'_data.csv', encoding='utf-8', sep='\t')
	df = pd.concat([df1, df2, df3]).sort_values('target_group')
	df.to_csv(savefile, index=False,  encoding='utf-8', sep='\t')


def dissect_temp(temp):
	'''
	Split template into separate words and remove target group placeholder.
	'''
	word_list = temp.split()
	word_list.remove("term_to_use")
	return word_list, " "+word_list[-1]+" "

def check_query(word_list, pred, group):
	'''
	Check if the query still has the correct form as the autocomplete suggestion API's can change the query input.	
	'''
	if (all(word in pred for word in word_list )) and (word_list[-1]+' ' in pred) and (group in pred):
		return True
	else:
		return False

def main(eng_target_dict, templates, country_templates, savefile='merged_data.csv'):
	'''
	Retrieve data from all three search engines into a csv file.
	'''
	data_collection(eng_target_dict, templates, country_templates, google, savefile ='google_data.csv')
	data_collection(eng_target_dict, templates, country_templates, yahoo, savefile ='yahoo_data.csv')
	data_collection(eng_target_dict, templates, country_templates, duckduckgo, savefile ='duckduckgo_data.csv')
	merge_data('google', 'yahoo', 'duckduckgo', savefile)


if __name__ == '__main__':
    main(eng_target_dict, templates, country_templates)