import random, pickle, os, sys
from datetime import datetime
import torch
import torch.nn as nn
from gensim.models import KeyedVectors
from sklearn.metrics import accuracy_score

from hyperparameters import Hyperparameters
import utils
from models import Encoder, Decoder, GenderClassifier
from losses import AdversarialLoss, ConfinementrLoss




hp = Hyperparameters()
torch.manual_seed(hp.torch_seed)
random.seed(hp.random_seed)






def run_classifier(inputs, labels, iteration, mode="train"):
    if mode == "train":
        classifiers[-1].train()
    else:
        classifiers[-1].eval()
    
    total_loss = 0
    total_num = 0

    y_pred = []
    y_true = []

    for i in range(len(inputs)):
        input = inputs[i].to(device)
        label = labels[i].to(device)

        if mode == "train":
            classifiers[-1].zero_grad()
            encoder.zero_grad()
            decoder.zero_grad()

        hidden_z, _ = encoder(input)
        prediction = classifiers[-1](hidden_z)
        loss = classification_criterion(prediction, label)

        if mode == "train":
            loss.backward()
            classifier_optimizers[-1].step()

        total_loss += loss.item()
        total_num += len(input)

        y_true += label.tolist()
        y_pred += torch.argmax(prediction, dim=1).tolist()

    return total_loss / total_num, accuracy_score(y_true, y_pred)







def run_autoencoder(inputs, labels, iteration, mode="train"):
    if mode == "train":
        encoder.train()
        decoder.train()
    else:
        encoder.eval()
        decoder.eval()

    total_loss = 0
    total_num = 0

    y_true = []
    y_pred = []

    for i in range(len(inputs)):
        input = inputs[i].to(device)
        label = labels[i].to(device)

        if mode == "train":
            encoder.zero_grad()
            decoder.zero_grad()
            for it in range(iteration):
                classifiers[it].zero_grad()

        hidden_z, hidden_g = encoder(input)
        reconstruction = decoder(torch.cat((hidden_z, hidden_g), dim=-1))

        reconstruction_loss = reconstruction_criterion(input, reconstruction)
        booster_loss = confinement_criterion(hidden_g.squeeze(), label.float())

        adversarial_loss = 0
        for it in range(iteration):
            prediction = classifiers[it](hidden_z)
            adversarial_loss += adversary_criterion(prediction)
            

        sum_lambdas = hp.lambda_reconstruction + hp.lambda_adversary + hp.lambda_booster

        loss = (hp.lambda_reconstruction / sum_lambdas) * reconstruction_loss + (hp.lambda_adversary / sum_lambdas) * adversarial_loss + (hp.lambda_booster / sum_lambdas) * booster_loss
        

        if mode == "train":
            loss.backward()
            encoder_optimizer.step()
            decoder_optimizer.step()

        # Compute the loss
        total_loss += loss.item()
        total_num += len(input)


        y_true += label.tolist()
        y_pred += torch.argmax(prediction, dim=1).tolist()
    
    return total_loss / total_num, accuracy_score(y_true, y_pred)


            












if __name__ == "__main__":

    # Checking the usage of GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() and hp.gpu else "cpu")
    print(device)

    
    # Preparing the data
    print("Loading word embeddings")
    embeddings = KeyedVectors.load_word2vec_format(hp.word_embedding, binary=hp.emb_binary)

    # Load wordlists
    print("Loading the wordlists")
    male_wordlist = utils.remove_oov_words(hp.male_words, embeddings)
    female_wordlist = utils.remove_oov_words(hp.female_words, embeddings)
    genderless_wordlist = utils.remove_oov_words(hp.neutral_words, embeddings)
    stereotype_female_wordlist, stereotype_male_wordlist = utils.remove_oov_words_stereotype(hp.stereotype_words, embeddings)

    # Prepare the labels for each word
    # 0 for female, 1 for genderless and 2 for male
    words = male_wordlist + stereotype_male_wordlist + female_wordlist + stereotype_female_wordlist + genderless_wordlist
    labels = [2] * (len(male_wordlist) + len(stereotype_male_wordlist)) + [0] * (len(female_wordlist) + len(stereotype_female_wordlist)) + [1] * len(genderless_wordlist)


    # Shuffle the words and the labels in the same way
    tmp = list(zip(words, labels))
    random.shuffle(tmp)
    words, labels = zip(*tmp)

    # Transform the words from strings to embeddings
    words = [embeddings[w] for w in words]

    # Split into train and eval
    train_data = torch.split(torch.tensor(words[hp.dev_num:]), hp.batch_size)
    test_data = torch.split(torch.tensor(words[:hp.dev_num]), hp.batch_size)

    train_labels = torch.split(torch.tensor(labels[hp.dev_num:]), hp.batch_size)
    test_labels = torch.split(torch.tensor(labels[:hp.dev_num]), hp.batch_size)



    print("Starting the pretrained models")
    # Load the autoencoder
    encoder = Encoder(hp.embedding_dim, hp.hidden_dim, hp.latent_dim - hp.gender_dim, hp.gender_dim, hp.dropout)
    decoder = Decoder(hp.latent_dim, hp.hidden_dim, hp.embedding_dim)


    if hp.pretrain_autoencoder:
        print('Loading pretrained autoencoder')
        checkpoint = torch.load(hp.pretrained_autoencoder_filepath) 
        autoencoder_hp = checkpoint['hp']
        assert hp.hidden_dim == autoencoder_hp.hidden_dim
        assert hp.latent_dim == autoencoder_hp.latent_dim       
        encoder.load_state_dict(checkpoint['encoder'])
        decoder.load_state_dict(checkpoint['decoder'])

    # Load the adversary discriminator
    classifiers = []

    # Pushing the models to GPU
    encoder.to(device)
    decoder.to(device)

    # Instantiating the optimizers
    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=hp.lr_autoencoder)
    decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=hp.lr_autoencoder)
    classifier_optimizers = []
    
    # Computing class weights for Cross Entropy Loss
    class_counts = [len(female_wordlist) + len(stereotype_female_wordlist), len(genderless_wordlist), len(male_wordlist) + len(stereotype_male_wordlist)]
    inverted_class_counts = [1/x for x in class_counts]
    class_weights = torch.FloatTensor([x/sum(inverted_class_counts) for x in inverted_class_counts]).to(device)


    
    classification_criterion = nn.CrossEntropyLoss(reduction="sum", weight=class_weights)
    reconstruction_criterion = nn.MSELoss(reduction="sum")
    adversary_criterion = AdversarialLoss(num_classes=hp.num_classes, reduction="sum", device=device)
    confinement_criterion = ConfinementrLoss(reduction="sum")

    hp.save_model = hp.save_model + str(datetime.now()).split(".")[0] + "/"
    try:
        os.mkdir(hp.save_model)
        with open(hp.save_model + "hp.json", 'w') as json_file:
            json_file.write(hp.toJSON())
    except OSError:
        print ("Creation of the directory %s failed" % hp.save_model)


    for it in range(1, 1 + hp.num_iterations):

        classifiers.append(GenderClassifier(hp.latent_dim - hp.gender_dim, hp.classification_hidden_dim, hp.num_classes))
        classifiers[-1].to(device)
        classifier_optimizers.append(torch.optim.Adam(classifiers[-1].parameters(), lr=hp.lr_classifier))



        print("Classification")
        for epoch in range(1, hp.num_epochs_classifier + 1):
            train_loss, train_acc = run_classifier(train_data, train_labels, it, mode="train")
            eval_loss, eval_acc = run_classifier(test_data, test_labels, it, mode="eval")

            if epoch % hp.print_every == 0:
                print("It {:02d} | Training {:.2f}% --> Train Loss = {} | Acc = {}".format(it, (epoch / hp.num_epochs_classifier) * 100, train_loss, train_acc))
                print("It {:02d} | Training {:.2f}% --> Test  Loss = {} | Acc = {}".format(it, (epoch / hp.num_epochs_classifier) * 100, eval_loss, eval_acc))
                print()

        
        print("Autoencoder")
        for epoch in range(1, hp.num_epochs_autoencoder + 1):
            train_loss, train_acc = run_autoencoder(train_data, train_labels, it, mode="train")
            eval_loss, eval_acc = run_autoencoder(test_data, test_labels, it, mode="eval")

            if epoch % hp.print_every == 0:
                print("It {:02d} | Training {:.2f}% --> Train Loss = {} | Acc = {}".format(it, (epoch / hp.num_epochs_autoencoder) * 100, train_loss, train_acc))
                print("It {:02d} | Training {:.2f}% --> Test  Loss = {} | Acc = {}".format(it, (epoch / hp.num_epochs_autoencoder) * 100, eval_loss, eval_acc))
                print()

        print()
        print("#############################")
        print()

    
        checkpoint = {
            'classifier': classifiers[-1].state_dict(),
            'encoder': encoder.state_dict(),
            'decoder': decoder.state_dict(),
            'hp': hp,
            'iterations': hp.num_iterations
        }
        torch.save(checkpoint, "{}/checkpoint{}".format(hp.save_model, it))
        if it > 1:
            os.remove('{}/checkpoint{}'.format(hp.save_model, it - 1))

        if it in hp.save_at:
            torch.save(checkpoint, "{}/{}.pt".format(hp.save_model, it))

    os.remove('{}/checkpoint{}'.format(hp.save_model, hp.num_iterations))


    
