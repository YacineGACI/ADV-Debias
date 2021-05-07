import torch
import torch.nn as nn 



class ConfinementrLoss(nn.Module):
    def __init__(self, reduction="mean"):
        super(ConfinementrLoss, self).__init__()
        self.mse_loss = nn.MSELoss(reduction=reduction)

    def forward(self, predictions, labels):
        labels = labels - 1 # Expects 0 for female, 1 for neutral and 2 for male
                            # Then, -1 for female, 0 for neutral and 1 for male
        return self.mse_loss(predictions, labels)




class AdversarialLoss(nn.Module):
    def __init__(self, num_classes=3, reduction="mean", device=torch.device("cpu")):
        super(AdversarialLoss, self).__init__()
        self.kl_div = nn.KLDivLoss(reduction=reduction)
        self.log_softmax = nn.LogSoftmax(dim=-1)
        self.num_classes = num_classes
        self.device = device

    def forward(self, predictions):
        log_softmax_logits = self.log_softmax(predictions)
        uniform = torch.tensor([[1/self.num_classes] * self.num_classes] * predictions.shape[0]).to(self.device)
        return self.kl_div(log_softmax_logits, uniform)

