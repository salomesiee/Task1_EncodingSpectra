import torch.nn.functional as F
import torch.nn as nn 
import torch 


class CLIPLoss(nn.Module):
    """ Ideal in the cas where there are no same label / couples in a batch """
    def __init__(self, init_tau=0.07):
        super().__init__()
        self.log_tau = nn.Parameter(torch.tensor(torch.log(torch.tensor(1.0 / init_tau))))

    def forward(self, out_ftir, out_raman, labels=0):
        tau = torch.exp(self.log_tau).clamp(max=100)
        logits = (out_ftir @ out_raman.T) * tau
        targets = torch.arange(len(logits), device=logits.device).long()

        raman_loss = F.cross_entropy(logits, targets)
        ftir_loss = F.cross_entropy(logits.T, targets)
        loss = (ftir_loss + raman_loss) / 2.0  
        return loss

class ConstrastiveLoss(nn.Module):
    def __init__(self, init_tau=0.07):
        super().__init__()
        self.log_tau = nn.Parameter(torch.tensor(torch.log(torch.tensor(1.0 / init_tau))))

    def forward(self, out_ftir, out_raman, labels):
        tau = torch.exp(self.log_tau).clamp(max=100)
        logits = (out_ftir @ out_raman.T) * tau                                                                                                                                                                                                                        
        
        labels = labels.view(-1, 1)
        positive_mask = (labels == labels.T).float()
        targets = positive_mask / positive_mask.sum(dim=1, keepdim=True)

        raman_loss = F.cross_entropy(logits, targets)
        ftir_loss = F.cross_entropy(logits.T, targets)
        loss = (ftir_loss + raman_loss) / 2.0  
        return loss


class SimilarityLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, out_ftir, out_raman, labels):
        similarity = F.cosine_similarity(out_ftir, out_raman)
        loss = 1 - similarity.mean()
        return loss

class SymmetricKL(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, out_ftir, out_raman, labels):
        p = F.softmax(out_ftir, dim=-1)
        q = F.softmax(out_raman, dim=-1)
        kl_pq = F.kl_div(p.log(), q, reduction='batchmean')
        kl_qp = F.kl_div(q.log(), p, reduction='batchmean')
        loss = (kl_pq + kl_qp) / 2
        return loss