import torch
import torch.nn as nn



class Encoder(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, z_dim, g_dim, dropout=0.1):
        super(Encoder, self).__init__()
        self.linear1 = nn.Linear(embedding_dim, hidden_dim)
        self.linear_z = nn.Linear(hidden_dim, z_dim)
        self.linear_g = nn.Linear(hidden_dim, g_dim)
        self.activation = nn.Tanh()
        self.dropout = nn.Dropout(dropout)


    def forward(self, x):
        hidden = self.activation(self.linear1(self.dropout(x)))
        z = self.linear_z(self.dropout(hidden))
        g = self.linear_g(self.dropout(hidden))
        return z, g







class Decoder(nn.Module):
    def __init__(self, decoder_dim, hidden_dim, embedding_dim):
        super(Decoder, self).__init__()
        self.linear1 = nn.Linear(decoder_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, embedding_dim)
        self.activation = nn.Tanh()


    def forward(self, x):
        hidden = self.activation(self.linear1(x))
        return self.linear2(hidden) # I use a linear activation function (identity) because the decoder
                                    # needs to reconstruct the original space







class GenderClassifier(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_labels=3, dropout=0.1):
        super(GenderClassifier, self).__init__()
        self.linear1 = nn.Linear(embedding_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, num_labels)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.softmax = None  # I don't need the softmax in the classifier, because CROSSENTROPYLOSS combines LogSoftmax and NLLLoss in one signle class

    def forward(self, x):
        return self.linear2(self.dropout(self.activation(self.linear1(x))))
        # Here, I don't need softmax, I apply it in the criterion (CrossEntropyLoss)






class GenderClassifier_Linear(nn.Module):
    def __init__(self, embedding_dim, num_labels=3):
        super(GenderClassifier_Linear, self).__init__()
        self.linear = nn.Linear(embedding_dim, num_labels)
        self.softmax = None  # I don't need the softmax in the classifier, because CROSSENTROPYLOSS combines LogSoftmax and NLLLoss in one signle class

    def forward(self, x):
        return self.linear(x)
        # Here, I don't need softmax, I apply it in the criterion (CrossEntropyLoss)