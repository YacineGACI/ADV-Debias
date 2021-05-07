import torch
from hyperparameters import Hyperparameters

hp = Hyperparameters()


def remove_oov_words(data_filepath, embeddings):
    # Returns the list of words in @data_filepath that belong to the vocabulary of @embeddings
    wordlist = []
    with open(data_filepath, 'r') as f:
        for line in f.readlines():
            for word in line.strip().split('\t'):
                if word in embeddings:
                    wordlist.append(word)
    return wordlist



def remove_oov_words_stereotype(data_filepath, embeddings):
    # Returns the list of words in @data_filepath that belong to the vocabulary of @embeddings
    female_list = []
    male_list = []
    with open(data_filepath, 'r') as f:
        for line in f.readlines():
            for pair in line.strip().split('\t'):
                if pair[0] in embeddings:
                    female_list.append(pair[0])
                if pair[1] in embeddings:
                    male_list.append(pair[1])
    return female_list, male_list
    