import random, os
from datetime import datetime
import torch
import torch.nn as nn
from gensim.models import KeyedVectors

from hyperparameters import PretrainingHyperparameters
import utils
from models import Encoder, Decoder




hp = PretrainingHyperparameters()
torch.manual_seed(hp.torch_seed)
random.seed(hp.random_seed)





def run_autoencoder(inputs, mode="train"):
    if mode == "train":
        encoder.train()
        decoder.train()
    else:
        encoder.eval()
        decoder.eval()

    total_loss = 0
    total_num = 0

    for i in range(len(inputs)):
        input = inputs[i].to(device)

        if mode == "train":
            encoder.zero_grad()
            decoder.zero_grad()

        hidden_z, hidden_g = encoder(input)
        mask = torch.randint(0, 2, hidden_g.shape).to(device)
        hidden_g = hidden_g * mask
        reconstruction = decoder(torch.cat((hidden_z, hidden_g), dim=-1))

        loss = reconstruction_criterion(input, reconstruction)
        

        if mode == "train":
            loss.backward()
            encoder_optimizer.step()
            decoder_optimizer.step()

        total_loss += loss.item()
        total_num += len(input)
    
    return total_loss / total_num










if __name__ == "__main__":

    # Checking the usage of GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() and hp.gpu else "cpu")
    print(device)

    
    # Preparing the data
    print("Loading word embeddings")
    embeddings = KeyedVectors.load_word2vec_format(hp.word_embedding, binary=hp.emb_binary)
    embeddings = [embeddings[word] for word in embeddings.wv.vocab]

    # Shuffle the data
    print("Shuffling the data")
    random.shuffle(embeddings)

   

    # Split into train and eval
    print("Splitting into train and test")
    train_data = torch.split(torch.tensor(embeddings[hp.pta_dev_num:]), hp.batch_size)
    test_data = torch.split(torch.tensor(embeddings[:hp.pta_dev_num]), hp.batch_size)



    # Load the autoencoder
    print("Starting the pretrained models")
    encoder = Encoder(hp.embedding_dim, hp.hidden_dim, hp.latent_dim - hp.gender_dim, hp.gender_dim, hp.dropout)
    decoder = Decoder(hp.latent_dim, hp.hidden_dim, hp.embedding_dim)


    # Pushing the models to GPU
    encoder.to(device)
    decoder.to(device)


    # Instantiating the optimizers
    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=hp.lr_autoencoder, weight_decay=hp.wd_autoencoder)
    decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=hp.lr_autoencoder, weight_decay=hp.wd_autoencoder)
    
    
    
    reconstruction_criterion = nn.MSELoss(reduction="sum")


    hp.save_model = hp.save_model + str(datetime.now()).split(".")[0] + "/"
    try:
        os.mkdir(hp.save_model)
        with open(hp.save_model + "hp.json", 'w') as json_file:
            json_file.write(hp.toJSON())
    except OSError:
        print ("Creation of the directory %s failed" % hp.save_model)



    
    
    print("Pretraining Autoencoder")
    for epoch in range(1, hp.num_epochs_autoencoder + 1):
        train_loss = run_autoencoder(train_data, mode="train")
        eval_loss = run_autoencoder(test_data, mode="eval")

        if epoch % hp.print_every == 0:
            print("Training {:.2f}% --> Train Loss = {}".format((epoch / hp.num_epochs_autoencoder) * 100, train_loss))
            print("Training {:.2f}% --> Test  Loss = {}".format((epoch / hp.num_epochs_autoencoder) * 100, eval_loss))
            print()

        
        checkpoint = {
            'encoder': encoder.state_dict(),
            'decoder': decoder.state_dict(),
            'hp': hp,
            'epoch': epoch
        }
        torch.save(checkpoint, "{}/autoencoder.pt".format(hp.save_model))


