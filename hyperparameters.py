import json

class JsonifiableObject:
    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__, 
            sort_keys=True, indent=4)










class Hyperparameters(JsonifiableObject):

    def __init__(self):
        
        # Filepaths to different word lists
        self.male_words = "data/wordlist/male_word_file.txt"
        self.female_words = "data/wordlist/female_word_file.txt"
        self.neutral_words = "data/wordlist/no_gender_list.tsv"
        self.stereotype_words = "data/wordlist/stereotype_list.tsv"

        self.save_model = "saved/trained/"

        self.word_embedding = 'embeddings/glove.txt'
        self.emb_binary = False

        self.embedding_dim = 300
        self.hidden_dim = 100
        self.latent_dim = 300
        self.gender_dim = 1


        self.classification_hidden_dim = 100
        self.num_classes = 3

        
        self.num_epochs_autoencoder = 5000
        self.num_epochs_classifier = 2000
        self.lr_autoencoder = 1e-6
        self.lr_classifier = 1e-5
        self.wd_autoencoder = 1e-5
        self.wd_classifier = 1e-5
        self.dev_num = 300
        self.batch_size = 512
        self.dropout = 0.2


        self.lambda_reconstruction = 1.0
        self.lambda_adversary = 0.9
        self.lambda_booster = 0.9

        self.num_iterations = 40
        self.save_at = [1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 35, 40]

        self.print_every = 200 

        self.gpu = True
        self.pretrain_autoencoder = True
        self.pretrained_autoencoder_filepath = "saved/pretrained/2021-05-07 17:11:55/autoencoder.pt"

        self.random_seed = 0
        self.torch_seed = 0






class PretrainingHyperparameters(Hyperparameters):

    def __init__(self):
        super(PretrainingHyperparameters, self).__init__()

        self.save_model = "saved/pretrained/"
        self.num_epochs_autoencoder = 315
        self.lr_autoencoder = 0.0002
        self.wd_autoencoder = 1

        self.pta_dev_num = 5000
        self.dropout = 0.05

        self.print_every = 1

        self.gpu = "True"