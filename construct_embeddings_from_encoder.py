import random, math, tqdm, pickle, sys

import torch
import torch.nn as nn

from gensim.models import KeyedVectors

from models import Encoder, Decoder






if __name__ == "__main__":

    filepath = sys.argv[1]
    num_iterations = int(sys.argv[2])

    # Txt file where to store the new embeddings
    output_filename = "embeddings/new/{}_{}.txt".format(filepath, num_iterations)

    ### Load the trained model
    checkpoint = torch.load('saved/trained/{}/{}.pt'.format(filepath, num_iterations))
    hp = checkpoint['hp']
    

    ### Read the input embeddings
    input_embeddings_filename = "embeddings/glove.txt"
    embeddings_vectors = KeyedVectors.load_word2vec_format(input_embeddings_filename, binary=False)
    print("Embeddings vectors read")
    

    ### Define the autoencoder
    encoder = Encoder(hp.embedding_dim, hp.hidden_dim, hp.latent_dim - hp.gender_dim, hp.gender_dim, hp.dropout)
    encoder.load_state_dict(checkpoint['encoder'])
    encoder.eval()



    ### Load the words to debias
    to_debias = []
    with open("data/debias.pkl", "rb") as f:
        to_debias = pickle.load(f)



    with open(output_filename, 'w') as f:

        f.write("322636 300\n")

        words = list(embeddings_vectors.wv.vocab)

        for w in tqdm.tqdm(to_debias):
            # For these words, make the gender dimensions equal to 0
            hidden_z, hidden_g = encoder(torch.tensor(embeddings_vectors[w]).unsqueeze(0))
            new_embedding = torch.cat((hidden_z, torch.zeros(hidden_g.shape)), dim=-1).squeeze().tolist()
            f.write("{} {}\n".format(w, " ".join([str(round(x, 8)) for x in new_embedding])))

        genderful_words = list(set(words) - set(to_debias))
        for w in tqdm.tqdm(genderful_words):
            # For the remaining words, keep the gender dimension as is
            hidden_z, hidden_g = encoder(torch.tensor(embeddings_vectors[w]).unsqueeze(0))
            new_embedding = torch.cat((hidden_z, hidden_g), dim=-1).squeeze().tolist()
            f.write("{} {}\n".format(w, " ".join([str(round(x, 8)) for x in new_embedding])))
