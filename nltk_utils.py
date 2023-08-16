import nltk
import numpy as np
nltk.download('punkt')
from nltk.stem.porter import PorterStemmer  #PorterStemmer est un algorithme de stemming qui tente de transformer les mots en leur forme racine en suivant un ensemble de règles spécifiques.
stemmer = PorterStemmer()

def tokenize(sentence):
    return nltk.word_tokenize(sentence)

def stem(word):
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence, all_words):
    """
        sentence = ["hello" ,"how","are","you"]
        words = ["hi","hello","I","you","bye","thank","cool"]
        bag = [0, 1, 0, 1, 0, 0, 0]

    """
    tokenized_sentence =[stem(w) for w in tokenized_sentence]
    bag = np.zeros(len(all_words), dtype=np.float32) #dtype=datatype
    for idx, w in enumerate(all_words): #enum give the val & the position in an array
        if w in tokenized_sentence:
            bag[idx]=1.0

    return bag

sentence = ["hello" ,"how","are","you"]
words = ["hi","hello","I","you","bye","thank","cool"]
bag = bag_of_words(sentence, words)
print(bag)
