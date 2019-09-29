# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 10:50:06 2019

@author: saurabh.patel
"""

import os
import string
import xml.etree.ElementTree as ET

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
                        
                        img_findings[image_id]=(f.text)
                        
                    # image and comparison list    
                    if f.get('Label') == 'COMPARISON':
                        
                        img_comparison[image_id]=(f.text)
                        
                    # image and indication list    
                    if f.get('Label') == 'INDICATION':
                        
                        img_indication[image_id]=(f.text)
                        
                    # image and impressions list    
                    if f.get('Label') == 'IMPRESSION':
                        
                        img_impression[image_id]=(f.text)
                        
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

def to_vocabulary(descriptions):
    desc = set()
    for key in descriptions.keys():
        d = descriptions[key]
        if not d[0] is None:
            #print(key)      
            [desc.update(d.split()) for d in descriptions[key]]
    return desc

## vocab is a set of impression findings and indication as words
vocab_impression = to_vocabulary(impression)
vocab_findings = to_vocabulary(findings)
vocab_indication = to_vocabulary(indication)

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

import pandas as pd

df_imp = pd.DataFrame(list(impression.items()),columns = ['image_id','impression'])


df_findings = pd.DataFrame()

for key,value in findings.items():
    df_findings['image_id'] = key
    df_findings['findings'] = value[0]




