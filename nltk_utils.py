import nltk
nltk.download('punkt')
from nltk.stem.porter import PorterStemmer  #PorterStemmer est un algorithme de stemming qui tente de transformer les mots en leur forme racine en suivant un ensemble de règles spécifiques.
stemmer = PorterStemmer()

def tokenize(sentence):
    return nltk.word_tokenize(sentence)

def stem(word):
    return stemmer.stem(word.lower())

def bag_of_words(tokenize_sentence, all_words):
    pass

a = "How long does shipping take?"
print(a)
a = tokenize(a)
print(a)

words = ["organize", "organizes","organizing"]
print(words)
stemmed_words=[stem(w) for w in words]
print(stemmed_words)