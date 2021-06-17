from transformers import AutoModelForMaskedLM, AutoTokenizer, pipeline
import torch
import pandas as pd 
import numpy as np 
import spacy 
import json
from nrclex import NRCLex
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import groupby
from collections import defaultdict
import os.path


dictionary1 = json.load(open('new_result/roberta-base-corrected.json',"r"))
dictionary2 = json.load(open('new_result/roberta-base.json',"r"))
for i in dictionary1.keys():
	for j in dictionary1[i][:10]:
		print("Corrected:", j[0])

	for j in dictionary2[i][:10]:
		print("Original:", j[0])