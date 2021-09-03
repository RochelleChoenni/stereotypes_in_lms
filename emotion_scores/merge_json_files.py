import os
import json 

def merge_json(path, source):

    full_dict = {}
    for filename in os.listdir(path):
        if filename.endswith(".json") and source in filename: 
            dictionary = json.load(open(path+filename,"r"))
            full_dict.update(dictionary)
        else:
            continue

    with open(path+'compress/'+source+'.json', 'w') as outfile:
        json.dump(full_dict, outfile)
