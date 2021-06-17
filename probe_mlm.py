# Import packages
import json
import torch
import logging
import itertools
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from torch.nn import functional as F
from transformers import AutoModelForMaskedLM, AutoTokenizer, pipeline

# Import files
from utils.utils import accuracy_at_n
from utils.target_dicts import eng_target_dict


def get_pred_completion(dataset, model_name, save_name):
	'''
	Get the top 200 model predictions for completing the template sentences from the
	search engine dataset.
	'''
	model, tokenizer = load_model(model_name)
	df = pd.read_csv(dataset,  encoding='utf-8', sep='\t')

	succes, total = 0, 0
	retrieved = []
	out_dict= {}
	for index, row in df.iterrows():

		# Construct and tokenize input sequence
		sequence = tokenizer.cls_token + " " +  row['input'].strip() + " " + tokenizer.mask_token
		sequence = sequence + ' ? ' + tokenizer.sep_token
		input = tokenizer.encode(sequence, return_tensors="pt")
		# Get masked token index
		mask_token_index = torch.where(input == tokenizer.mask_token_id)[1]

		# Get model output
		token_logits = model(input)[0]
		mask_token_logits = token_logits[0, mask_token_index, :]
		sm = torch.nn.Softmax(dim=0)

		# Sort top 200 output by language model probability
		top_k_weights, top_k_tokens = torch.topk(mask_token_logits, 200, dim=1)
		weights = top_k_weights[0]
		weights = sm(weights).tolist()
		top_k_tokens =  top_k_tokens[0].tolist()
		
		# Get top 200 output
		output, t = [], 0
		for token in top_k_tokens:
			pred = tokenizer.decode([token])
			output.append((pred, weights[t]))
			t+=1

			# Test how many of the MLM predictions are identical to ground truth completion
			if row['completion'].strip() == pred.strip():
				logging.info("Correct: ", row['completion'].strip())
				succes +=1
				retrieved.append(sequence.replace(tokenizer.mask_token, pred))

		total +=1

		if row['input'].strip() in out_dict.keys():
			out_dict[row['input'].strip()].extend(output)
		else:
			out_dict[row['input'].strip()] = output

	logging.info("Succesfully retrieved: ", succes, ' out of ', total) 

	o = json.dumps(out_dict)
	f = open(save_name+"_dict.json","w")
	f.write(o)
	f.close()


def get_prior_probs(model_name, save_name, completion_dict):
	'''
	Get prior probabilities of attributes given any (masked) target group.
	'''
	model, tokenizer = load_model(model_name)

	# Construct input sequence
	sentence = f"Why are {tokenizer.mask_token} so {tokenizer.mask_token}"
	token_ids = tokenizer.encode(sentence, return_tensors='pt')
	masked_position = torch.where(token_ids[0] == tokenizer.mask_token_id)[0]

	# Get prior outputs
	output = model(token_ids)
	last_hidden_state = output[0].squeeze(0)
	# Only get output for masked token for stereotypical attribute
	mask_hidden_state = last_hidden_state[masked_position[1]]

	# Convert to probabilities (softmax)
	sm = torch.nn.Softmax(dim=0)	
	probs = sm(mask_hidden_state)

	# Get all autocmpletions
	dictionary = json.load(open(completion_dict,"r"))
	flatten = itertools.chain.from_iterable
	dictionary = list(flatten(list(dictionary.values())))
	terms = list(set([i[0].strip() for i in dictionary]))

	prior_probs = {}

	# Get probability for each autocompletion 
	for term in terms:
		term_id = tokenizer.convert_tokens_to_ids(term)
		prob = probs[term_id].item()
		prior_probs[term] = prob

	prior_probs_list = sorted(prior_probs, key=prior_probs.get, reverse=True)
	o = json.dumps(prior_probs)
	f = open(save_name+"_priors.json","w")
	f.write(o)
	f.close()

	return prior_probs


def reranking_with_prior_probs(model_name, save_name, completion_dict, prior_dict):
	'''
	Re-rank top k predictions based on their stereotypicality.
	'''
	prior_dict = json.load(open(prior_dict,"r"))
	dictionary = json.load(open(completion_dict,"r"))

	new_dict = {}
	for i in dictionary.keys():
		new = []
		for c in dictionary[i]:
			new.append([c[0].strip(), np.log(c[1]/prior_dict[c[0].strip()])])
		new = [list(x) for x in set(tuple(x) for x in new)]
		new.sort(key = lambda x: x[1], reverse=True) 
		new_dict[i] = new

	o = json.dumps(new_dict)
	f = open(save_name+"-corrected.json","w")
	f.write(o)
	f.close()


def compute_recall_scores(name, cat, n, data_path='data_collection/stereo_dataset/final_probe.csv'):
	'''
	Compute recall scores over the number of stereotypical attributes in the stereodataset
	also encoded in the pretraned model. 
	'''
	compl_dict=json.load(open(name,"r"))
	df = pd.read_csv(data_path,  encoding='utf-8', sep='\t')
	recall = 0
	total = 0
	for i in compl_dict.keys():
		cats = [w.lower() for w in eng_target_dict[cat]]
		target = df.loc[df['input'] == i+'\n', 'target_group'].values
		if target[0].strip() in cats:
			compls = df[df['input']==i+'\n']['completion'].values
			compls = [i.strip() for i in compls]
			preds = [p[0] for p in compl_dict[i]]
			scores =[accuracy_at_n(preds, c, n) for c in compls]
			retrieved = [compls[i] for i in range(0, len(scores)) if scores[i] == 1 ]
			recall+=np.sum(scores)
			total += len(scores)

	logging.info('Recall at ', n, ' is ', recall, '/', total, ' -- ', recall/total)

	return recall/total

def load_model(model_name):
	# Load pretrained model and corresponding tokenizer
	if model_name == 'bart-base' or model_name == 'bart-large': 
		tokenizer = AutoTokenizer.from_pretrained('facebook/'+model_name)
		model = AutoModelForMaskedLM.from_pretrained('facebook/'+model_name)
	else: 
		tokenizer = AutoTokenizer.from_pretrained(model_name)
		model = AutoModelForMaskedLM.from_pretrained(model_name)
	return model, tokenizer


def complete_mask(model_name):
	'''
	Mask both the target group and attribute retrieve the top k preferred 
	completions for the template sentence.
	'''
	sm = torch.nn.Softmax(dim=0)
	torch.set_grad_enabled(False)
	
	tokenizer = AutoTokenizer.from_pretrained(model_name)
	model = AutoModelForMaskedLM.from_pretrained(model_name)
	
	sentence = f"Why are {tokenizer.mask_token} so {tokenizer.mask_token} ?"
	token_ids = tokenizer.encode(sentence, return_tensors='pt')
	masked_position = torch.where(token_ids[0] == tokenizer.mask_token_id)[0]

	output = model(token_ids)
	last_hidden_state = output[0].squeeze(0)

	for k in range(100):
		preds = []
		for m in masked_position:
			sorted_preds, sorted_idx = last_hidden_state[m].sort(dim=-1, descending=True)
			predicted_index = sorted_idx[k].item()
			predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0] 
			preds.append(predicted_token)
		logging.info("Why are ", preds[0], ' so ', preds[1])


def evaluate(name, cat, n):
	compl_dict=json.load(open(name,"r"))
	df = pd.read_csv('./data_collection/stereo_dataset/single_word_stereo_dataset.csv',  encoding='utf-8', sep='\t')

	recall = 0
	total = 0
	for i in compl_dict.keys():
		cats = [w.lower() for w in eng_target_dict[cat]]
		df = df.replace('\n','', regex=True)
		target = df.loc[df['input'] == i, 'target_group'].values
		if target[0].strip() in cats: 
			compls = df[df['input']==i]['completion'].values
			compls = [i.strip() for i in compls]
			preds = [p[0] for p in compl_dict[i]]
			scores =[accuracy_at_n(preds, c, n) for c in compls]
			retrieved = [compls[i] for i in range(0, len(scores)) if scores[i] == 1 ]
			recall+=np.sum(scores)
			total += len(scores)
			if retrieved != [] :
				print(i, ', '.join(list(set(retrieved))))
	print('Recall at ', n, ' is ', recall, '/', total, ' -- ', np.round(recall/total,2) , '% \n')

	return recall/total

def get_results_per_cat(cat, print_at):
	models = ['bert-base-uncased', 'bert-large-uncased', 'roberta-base', 'roberta-large', 'bart-base', 
				'bart-large', 'bert-base-multilingual-uncased', 'xlm-roberta-base', 'xlm-roberta-large'] 
	score_at = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105,
				 110, 115, 120, 125,130,135, 140, 145, 150, 155, 160, 165, 170, 175, 180, 185, 190, 195, 200]
	names = ['BERT-B', 'BERT-L', 'RoBERTa-B', 'RoBERTa-L', 'BART-B', 'BART-L', 'mBERT', 'XLMR-B', 'XLMR-L']
	ind = 0
	cm = ['royalblue', 'darkblue','sandybrown', 'darkorange', 'seagreen', 'darkgreen',  'slategrey', 'crimson', 'darkred' ]
	stl = ['-', '--', '-', '--', '-', '--', '-', '-', '--']
	for m in models: 
		results_corrected = []
		for n in score_at : 
			recall = evaluate('mlm_output/'+m+'-corrected.json', cat, n)
			results_corrected.append(recall)
		plt.plot(score_at, results_corrected, linestyle = stl[ind], color=cm[ind],label=names[ind], linewidth=2)
		ind+=1
	 
	plt.legend( prop={'size': 15})
	plt.title(cat.capitalize()+' stereotypes recall at k', fontsize=20)
	plt.tight_layout()
	plt.xlabel('k', fontsize=20)
	plt.ylim([0,0.85])
	plt.ylabel('Recall', fontsize=20)
	plt.xticks(size = 20)
	plt.yticks(np.arange(0, 1, step=0.2),size = 20)
	plt.grid(True)
	plt.savefig('all-models-'+cat+'.pdf', bbox_inches='tight',pad_inches = 0 )
	plt.show()

def study_results_at_per_cat(m, cat, print_at):
	names = {'BERT-B': 'bert-base-uncased', 'BERT-L': 'bert-large-uncased', 'RoBERTa-B': 'roberta-base', 'RoBERTa-L': 'roberta-large',
                'BART-B': 'bart-base', 'BART-L': 'bart-large', 'mBERT': 'bert-base-multilingual-uncased', 'XLMR-B': 'xlm-roberta-base', 'XLMR-L': 'xlm-roberta-large'}
	recall = evaluate('mlm_output/'+names[m]+'-corrected.json', cat, print_at)
	 



def get_finetuned_results_per_cat(cat, models, names, diri):
	
	score_at = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100,
	 105, 110, 115, 120, 125,130,135, 140, 145, 150, 155, 160, 165, 170, 175, 180, 185, 190, 195, 200]
	
	ind = 0
	for m in models: 
		results_corrected = []
		for n in score_at : 
			recall = evaluate(diri+m+'-corrected.json', cat, n)
			results_corrected.append(recall)
		plt.plot(score_at, results_corrected, label=names[ind], linewidth=2)
		ind+=1
	 
	plt.legend( prop={'size': 15})
	plt.title(cat.capitalize()+' stereotypes recall at k', fontsize=18)
	plt.tight_layout()
	plt.xlabel('k', fontsize=18)
	plt.ylim([0,1])
	plt.ylabel('Recall', fontsize=18)
	plt.xticks(size = 18)
	plt.yticks(size = 18)

	plt.grid(True)
	plt.savefig(diri+cat+'.pdf', bbox_inches='tight',pad_inches = 0 )
	plt.show()


def get_finetuned_all_cat(models, names, pick):

	score_at = np.arange(20, 220, 20 )
	score_at = [5, 10] + score_at.tolist()
	print(score_at)
	ind = 0
	cm = ['crimson', 'steelblue', 'mediumseagreen', 'slategray', 'darkorange']
	for m in models: 
		results_corrected = []
		for n in score_at : 
			total = []
			for c in ['political', 'age', 'gender', 'religion', 'lifestyle', 'profession', 'country', 'race']:
				recall = evaluate(m+'-corrected.json', c, n)
				total.append(recall)
			results_corrected.append(np.mean(total))
		plt.plot(score_at, results_corrected, linestyle='--', label=names[ind], color= cm[ind], linewidth=2)
		ind+=1
	 
	plt.legend( prop={'size': 15})
	plt.title('Stereotypes recall at k '+'('+pick+')', fontsize=18)
	plt.tight_layout()
	plt.xlabel('k', fontsize=18)
	plt.ylim([0,0.7])
	plt.ylabel('Recall', fontsize=18)
	plt.xticks(size = 18)
	plt.yticks(size = 18)

	plt.grid(True)
	plt.savefig(pick+'-human-finetuned.pdf', bbox_inches='tight',pad_inches = 0 )
	plt.show()

'''
for source in [ 'fox']:

	model = 'emotion_affect/finetuned-tenth/bert-base/'+source
	save_name = 'tenth-finetuned/'+source
	get_pred_completion('emotion_affect/stereo_dataset/final_probe.csv', model, save_name)
	get_prior_probs(model, save_name, save_name+'_dict.json')
	correct_with_prior_probs(model, save_name, save_name+'_dict.json', save_name+'_priors.json')

models = ['finetuned/bert-base/breitbart',  'finetuned/bert-base/reuters', 'finetuned/bert-base/guardian', 'finetuned/bert-base/new_yorker', 'finetuned/bert-base/fox',
			'half-finetuned/bert-base/breitbart',  'half-finetuned/bert-base/reuters', 'half-finetuned/bert-base/guardian', 'half-finetuned/bert-base/new_yorker', 'half-finetuned/bert-base/fox',
			'quart-finetuned/bert-base/breitbart',  'quart-finetuned/bert-base/reuters', 'quart-finetuned/bert-base/guardian', 'quart-finetuned/bert-base/new_yorker', 'quart-finetuned/bert-base/fox',
				'tenth-finetuned/bert-base/breitbart',  'tenth-finetuned/bert-base/reuters', 'tenth-finetuned/bert-base/guardian', 'tenth-finetuned/bert-base/new_yorker', 'tenth-finetuned/bert-base/fox']
names = ['breitbart-100',  'reuters-100', 'guardian-100', 'new_yorker-100', 'fox-100', 'breitbart-50',  'reuters-50', 'guardian-50', 'new_yorker-50', 'fox-50', 
			'breitbart-25',  'reuters-25', 'guardian-25', 'new_yorker-25', 'fox-25', 
			'breitbart-10',  'reuters-10', 'guardian-10', 'new_yorker-10', 'fox-10']
'''
'''
pick = 'breitbart' 
models = ['new_result/bert-base-uncased',
	'finetuned/bert-base/'+pick, 
			'half-finetuned/bert-base/'+pick, 
			'quart-finetuned/bert-base/'+pick, 
				'tenth-finetuned/bert-base/'+pick]

names = ['BERT-B',pick+'-100',  pick+'-50', pick+'-25',  pick+'-10']
'''
'''
for c in ['religion','race', 'profession', 'gender', 'age', 'political', 'lifestyle', 'country']: 
	#get_finetuned_per_cat(c, models, names, 'quart-finetuned/')
	get_results_per_cat(c)
'''
#get_finetuned_all_cat(models, names, pick)

