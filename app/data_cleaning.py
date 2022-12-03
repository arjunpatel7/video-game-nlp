
import pandas as pd
import numpy as np
import re

from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer

from textacy.preprocessing.remove import accents, brackets, punctuation
from textacy.preprocessing.replace import numbers, urls
from textacy.preprocessing.normalize import whitespace

import os

def clean_page(page):
    # given a page, removes heading, newlines, tabs, etc
    page = re.sub("=+", "", page)
    page = page.replace("\n", "")
    page = page.replace("\t", "")
    page = accents(brackets(page))
    page = urls(page)

    return whitespace(page).lower()

def clean_sentences(s):
        
    pattern = r'[^A-Za-z0-9]+'
    page = re.sub(pattern, '', s)
    return s


  
ps = PorterStemmer()
def prepare_document(doc):
    # given a document, preprocesses and tokenizes it for tfidf

    # clean the document of misc symbols and headings, lowercase it
    doc = clean_page(doc)

    #tokenize by sentence and then by word
    sentences = sent_tokenize(doc)

    #remove punctuation
    sentences = [punctuation(s) for s in sentences]


    # stem every word
    sentences_and_words = [word_tokenize(s) for s in sentences]

    prepared_doc = []
    
    for sent in sentences_and_words:
        stemmed_sentences = []
        for word in sent:
            stemmed_sentences.append(ps.stem(word))
        cleaned_sentence = " ".join(stemmed_sentences)
        prepared_doc.append(cleaned_sentence)
    return " ".join(prepared_doc)


# small function to calculats cosine similarity of all pairs and store
def cosine_similarity(v1, v2):
    numerator = np.dot(v1, v2)
    denom = np.sqrt(np.sum(np.square(v1))) * np.sqrt(np.sum(np.square(v2)))

    return numerator/denom 


def cos_dicts(names, vects):

    #given a set of vectors, create a dict of dicts for cosine similarity
    # This dict of dict structure allows us to index directly into the pair we want
    # The first key will be our desired game
    # and the value for that key will be a dictionary of partner games

    # The inner key will be the second game we wish to seek, and its value will be cosine similarity to our first game

    d = {}
    for name, vect in zip(names, vects):
        cos_sim_by_vect = {}
        for n2, v2 in zip(names, vects):
            if n2 != name:
                cos_sim_by_vect[n2] = cosine_similarity(vect, v2)
        d[name] = cos_sim_by_vect
    return d

def retrieve_top_k_similar(n1, similarity_dict, k):
    inner_dict = similarity_dict[n1]
    # sort the dictionary by value, descending, then retrieve top k values
    return sorted(inner_dict.items(), reverse = True, key = lambda x: x[1])[:k]
