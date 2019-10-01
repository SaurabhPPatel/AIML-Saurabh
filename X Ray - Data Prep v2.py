# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 10:50:06 2019

@author: saurabh.patel
"""

import os
import string
import xml.etree.ElementTree as ET
from keras.preprocessing.text import Tokenizer

### function for xml extraction.
# input directory path where xml is stored

def xml_extract(path):
    img_findings = dict()
    img_comparison = dict()
    img_indication = dict()
    img_impression = dict()
    for file in os.listdir(path):
        parser = ET.XMLParser(encoding="utf-8")
        tree = ET.parse(file,parser=parser)
        root = tree.getroot()
        #all_images = []
       
        for i in root.iter('parentImage'):
            image_id = i.get('id')
           
            if not image_id is None:
                #all_images.append(image_id)
                for f in root.iter('AbstractText'):
                    # image and findings list
                    if f.get('Label') == 'FINDINGS':
                        finding = f.text
                        img_findings[image_id]=list()
                        img_findings[image_id].append(f.text)
                    # image and comparison list    
                    if f.get('Label') == 'COMPARISON':
                        comparison = f.text
                        img_comparison[image_id]=list()
                        img_comparison[image_id].append(f.text)
                    # image and indication list    
                    if f.get('Label') == 'INDICATION':
                        indication = f.text
                        img_indication[image_id]=list()
                        img_indication[image_id].append(f.text)
                    # image and impressions list    
                    if f.get('Label') == 'IMPRESSION':
                        impression = f.text
                        img_impression[image_id]=list()
                        img_impression[image_id].append(f.text)
    return(img_findings,img_comparison,img_indication,img_impression)
    

### clean descriptions
def clean_descriptions(descriptions):
	# prepare translation table for removing punctuation
    table = str.maketrans('', '', string.punctuation)
    for key, desc_list in descriptions.items():
        for i in range(len(desc_list)):
                print(i)
                desc = desc_list[i]
                #print(desc)
                if desc is not None:  
        			# tokenize
                    desc = desc.split()
                    print(desc)
        			# convert to lower case
                    desc = [word.lower() for word in desc]
        			# remove punctuation from each token
                    desc = [w.translate(table) for w in desc]
        			# remove hanging 's' and 'a'
                    desc = [word for word in desc if len(word)>1]
        			# remove tokens with numbers in them
                    desc = [word for word in desc if word.isalpha()]
        			# store as string
                    desc_list[i] =  ' '.join(desc)
                    



            
path = os.chdir('D:\Great Learning\Capstone\Chest Xray\Reports\ecgen-radiology')

findings,comparison,indication,impression = xml_extract(path)

clean_descriptions(impression)
clean_descriptions(indication)
clean_descriptions(findings)

### convert descriptions to vocabulary

##### check 
imp = impression['CXR1000_IM-0003-1001']
imp
imp[0]
### check

def to_vocabulary(descriptions):
    desc = set()
    for key in descriptions.keys():
        d = descriptions[key]
        if not d[0] is None:
            #print(key)      
            [desc.update(d.split()) for d in descriptions[key]]
    return desc

vocab_impression = to_vocabulary(impression)
vocab_findings = to_vocabulary(findings)
vocab_indication = to_vocabulary(indication)

print('impression vocab size  :',len(vocab_impression))
print('findings vocab size  :',len(vocab_findings))
print('indication vocab size  :',len(vocab_indication))



def save_txt(desc,filename):
    line = list()
    for key,value in desc.items():
        if not value[0] is None:
            line.append(key + ' ' + value[0])
            data = '\n'.join(line)
            file = open(filename,'w')
            file.write(data)
            file.close()
        
os.chdir('D:\Great Learning\Capstone\Chest Xray')

#os.curdir()

save_txt(impression,'impression.txt')

save_txt(findings,'findings.txt')

save_txt(indication,'indication.txt')

### list of images
image_ids = list(impression.keys())

### split images into train and test and validation samples

import random
random.shuffle(image_ids)

train_image_ids = image_ids[:5229]
bal_image_ids = image_ids[5229:]

test_image_ids = bal_image_ids[:1121]
val_image_ids = bal_image_ids[1121:]


# load doc into memory
def load_doc(filename):
	# open the file as read only
	file = open(filename, 'r')
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text


# load clean descriptions into memory
def load_clean_descriptions(filename, dataset):
	# load document
	doc = load_doc(filename)
	descriptions = dict()
	for line in doc.split('\n'):
		# split line by white space
		tokens = line.split()
		# split id from description
		image_id, image_desc = tokens[0], tokens[1:]
		# skip images not in the set
		if image_id in dataset:
			# create list
			if image_id not in descriptions:
				descriptions[image_id] = list()
			# wrap description in tokens
			desc = 'startseq ' + ' '.join(image_desc) + ' endseq'
			# store
			descriptions[image_id].append(desc)
	return descriptions

train_imp = load_clean_descriptions('impression.txt',train_image_ids)
train_fin = load_clean_descriptions('findings.txt',train_image_ids)
train_ind = load_clean_descriptions('indication.txt',train_image_ids)



test_imp = load_clean_descriptions('impression.txt',test_image_ids)
test_fin = load_clean_descriptions('findings.txt',test_image_ids)
test_ind = load_clean_descriptions('indication.txt',test_image_ids)


val_imp = load_clean_descriptions('impression.txt',val_image_ids)
val_fin = load_clean_descriptions('findings.txt',val_image_ids)
val_ind = load_clean_descriptions('indication.txt',val_image_ids)


### Find out the max length of report descriptions
def to_lines(descriptions):
    desc = list()
    for key in descriptions.keys():
        [desc.append(d) for d in descriptions[key]]
    return desc

def max_length(descriptions):
    lines = to_lines(descriptions)
    return max(len(d.split()) for d in lines)

max_len_fin = max_length(train_fin)

print(max_len_fin)

### create word index using keras tokenizer alternatively word to index conversion can be done
    
def create_tokenizer(descriptions):
    lines = to_lines(descriptions)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer

tokenizer = create_tokenizer(train_fin)

len(tokenizer.word_index)








