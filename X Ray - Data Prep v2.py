# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 10:50:06 2019

@author: saurabh.patel
"""

import os
import string
import xml.etree.ElementTree as ET
from keras.preprocessing.text import Tokenizer

#### Steps

# Extract XML
# clean descriptions
# create vocab
# save descriptions txt
# test train split
# load train test validation clean descriptions with startseq and end seq
# find out max lenght of descriptions
# create tokenizer or word index
# process image data
# sequence generator
# word embeddings



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

print("Max length of findings : ",max_len_fin)

### create word index using keras tokenizer alternatively word to index conversion can be done
    
def create_tokenizer(descriptions):
    lines = to_lines(descriptions)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer

tokenizer = create_tokenizer(train_fin)

print("No of tokens for findings : ", len(tokenizer.word_index))


#### Create photo features and save to features.pkl #####

# load inception v3 model and remove the last layer
from keras.applications.inception_v3 import InceptionV3
from keras.models import Model
from keras import Input
from keras import layers
from os import listdir
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import numpy as np
from keras.applications.inception_v3 import preprocess_input
from pickle import dump
from pickle import load

# extract features for all images
def extract_photo_features(directory):
    
    model = InceptionV3(weights='imagenet')
    model_inv3 = Model(model.input,model.layers[-2].output)
    print(model_inv3.summary())
    feature_vec = dict()
    
    for name in listdir(directory):
        filename = directory + '/' + name
        img = load_img(filename,target_size=(299,299))
        x = img_to_array(img)
        x = np.expand_dims(x,axis=0)
        x = preprocess_input(x)
        
        feature = model_inv3.predict(x,verbose=0)
        feature = np.reshape(feature,feature.shape[1])
        
        img_id = name.split('.')[0]
        feature_vec[img_id] = feature
        
    return feature_vec


# following function is used to load features to training and test set
def load_photo_features(filename,dataset):
    all_features=load(open(filename,'rb'))
    features = {k: all_features[k] for k in dataset}
    return features


        
directory = "D:\Great Learning\Capstone\Chest Xray\Images"
features = extract_photo_features(directory)
print("total extracted features  :  ",len(features))
dump(features,open('features.pkl','wb'))

train_features_fin = load_photo_features('features.pkl',train_fin)
test_features_fin = load_photo_features('features.pkl',test_fin)


#### Sequence generator

from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from numpy import array

def create_sequence(tokenizer,max_length,descriptions,photo_features,vocab_size):
    
    x1,x2,y = list(),list(),list()
    for key,desc_list in descriptions.items():
        for desc in desc_list:
            seq = tokenizer.texts_to_sequences([desc])[0]
            for i in range(1,len(seq)):
                in_seq,out_seq = seq[:i],seq[i]
                in_seq = pad_sequences([in_seq],maxlen = max_length)[0]
                out_seq = to_categorical([out_seq],num_classes=vocab_size)[0]
                
                x1.append(photo_features[key][0])
                x2.append(in_seq)
                y.append(out_seq)
                
    return array(x1),array(x2),array(y)
 

#### Word embeddings using GLOVE
embedding_dim=300
glove_dir = "D:\\Great Learning\\Glove"
#file = os.path.join(glove_dir,'glov.6B.300d.txt')
    
embeddings_index = {}
f = open('glove.6B.300d.txt',encoding='utf-8')

for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:],dtype='float32')
    embeddings_index[word] = coefs
    
f.close()

word_index = tokenizer.word_index
embedding_matrix = np.zeros((len(word_index)+1,embedding_dim))
for word, i in word_index.items():
    embedding_vector =  embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector



##### Define Captioning model
    
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding
from keras.layers import Dropout
from keras.layers.merge import add


    
def define_model_description(vocab_size,max_length,embedding_dim):
    inputs1 = Input(shape=(2048,))
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(256,activation='relu')(fe1)
    
    inputs2 = Input(shape=(max_length,))
    se1 = Embedding(vocab_size,embedding_dim,mask_zero=True,weights=embedding_matrix,trainable=False)(inputs2)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(256)(se2)
    
    decoder1 = add([fe2,se3])
    decoder2 = Dense(256,activation='relu')(decoder1)
    
    outputs = Dense(vocab_size,activation='softmax')(decoder2)
    model = Model(inputs=[inputs1,inputs2],outputs=outputs)
    model.compile(loss='categorical_crossentropy',optimizer='adam')
    print(model.summary())
    
    return model
    
#### End of Define captioning model



#### Create x1 x2 and y set for train and test

x1train,x2train,ytrain = create_sequence(tokenizer,max_len_fin,train_fin,train_features_fin,(len(word_index)+1))

x1test,x2test,ytest = create_sequence(tokenizer,max_len_fin,test_fin,test_features_fin,(len(word_index)+1))    


#### fit the LSTM model

from keras.callbacks import ModelCheckpoint

model_LSTM = define_model_description((len(word_index)+1),max_len_fin,embedding_dim)
filepath = 'model-ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5'
checkpoint = ModelCheckpoint(filepath,monitor='val_loss',verbose = 1,save_best_only=True,mode='min')
model_LSTM.fit([x1train,x2train],ytrain,epochs=50,verbose=2,callbacks=[checkpoint],validation_data=([x1test,x2test],ytest))


