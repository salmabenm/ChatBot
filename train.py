import json
from nltk_utils import tokenize, stem, bag_of_words
import numpy as np

import torch

with open('intents.json','r') as f :
    intents = json.load(f)

all_words = []
tags = []
xy = [] #patterns and tags
for intent in intents['intents']: #the only array in intents.json intents is the key the first one is all the file
    tag = intent['tag'] #tag is the key
    tags.append(tag)
    for pattern in intent['patterns']:
        w = tokenize(pattern) 
        all_words.extend(w) # I use extend because w it's also an array
        xy.append((w, tag))

ignore_words = ['?','!','.',',']
all_words = [stem(w) for w in all_words if w not in ignore_words]
all_words = sorted(set(all_words)) #set for remove duplicate elements 
tags = sorted(set(tags)) #give a unique lables
print(tags)

x_train = []
y_train = []
for (pattern_sentece, tag) in xy:
    bag = bag_of_words(pattern_sentece, all_words)
    x_train.append(bag)

    label = tags.index(tag)
    y_train.append(label) 

x_train = np.array(x_train)
y_train = np.array(y_train)