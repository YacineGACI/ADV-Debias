import random, pickle
from gensim.models import KeyedVectors
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import pandas as pd
from numpy import dot
from numpy.linalg import norm

import seaborn as sns

import utils



def similarity(w1, w2):
    return dot(w1, w2)/(norm(w1)*norm(w2))




if __name__ == "__main__":

    # The input word embedding file
    embedding_filename = "new/2021-05-07 17:27:53_1"

    # Loading Word Embeddings
    embeddings = KeyedVectors.load_word2vec_format("embeddings/{}.txt".format(embedding_filename), binary=False)
    print("Word Embeddings loaded successfully")


    # Loading word lists and sort them
    male_wordlist = sorted(utils.remove_oov_words("data/wordlist/male_word_file.txt", embeddings))
    female_wordlist = sorted(utils.remove_oov_words("data/wordlist/female_word_file.txt", embeddings))
    genderless_wordlist = sorted(utils.remove_oov_words("data/wordlist/no_gender_list.tsv", embeddings))
    stereotype_wordlist = sorted(utils.remove_oov_words("data/wordlist/stereotype_list.tsv", embeddings))
    print("Word lists loaded and sorted succesfully")


    # Computing the gender dimension
    gender_vector = embeddings['he'] - embeddings['she']


    # Computing similarities of words with the gender dimension
    male_sim = []
    female_sim = []
    genderless_sim = []
    stereotype_sim = []

    for w in male_wordlist:
        male_sim.append(similarity(gender_vector, embeddings[w]))

    for w in female_wordlist:
        female_sim.append(similarity(gender_vector, embeddings[w]))

    for w in genderless_wordlist:
        genderless_sim.append(similarity(gender_vector, embeddings[w]))

    for w in stereotype_wordlist:
        stereotype_sim.append(similarity(gender_vector, embeddings[w]))


    dataframe = [[male_wordlist[i], male_sim[i], i,  "male-oriented"] for i in range(len(male_sim))]
    dataframe += [[female_wordlist[i], female_sim[i], len(male_sim) + i,  "female-oriented"] for i in range(len(female_sim))]
    dataframe += [[genderless_wordlist[i], genderless_sim[i], len(male_sim) + len(female_sim) + i,  "gender-neutral"] for i in range(len(genderless_sim))]
    dataframe += [[stereotype_wordlist[i], stereotype_sim[i], len(male_sim) + len(female_sim) + len(genderless_sim) + i,  "stereotype"] for i in range(len(stereotype_sim))]

    dataframe = pd.DataFrame(dataframe, columns=['word', 'similarity', 'rank', 'category'])
    

    sns.scatterplot(data=dataframe, x='similarity', y='rank', hue='category')

    ax = plt.gca()
    ax.axes.xaxis.set_visible(True)
    ax.axes.yaxis.set_visible(False)

    plt.xlim(-1, 1)
    plt.grid(False)


    plt.savefig("visualizations/figures/{}.png".format(embedding_filename))
    plt.show()