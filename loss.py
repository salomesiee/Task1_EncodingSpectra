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
        
        if False:
            labels = labels.view(-1, 1)
            positive_mask = (labels == labels.T).float()
            targets = positive_mask / positive_mask.sum(dim=1, keepdim=True)

        if True:
            ftir_sim = out_ftir @ out_ftir.T
            raman_sim = out_raman @ out_raman.T
            targets = F.softmax(
                (ftir_sim + raman_sim) / 2, dim=-1
            )

        raman_loss = F.cross_entropy(logits, targets)
        ftir_loss = F.cross_entropy(logits.T, targets)
        loss = (ftir_loss + raman_loss) / 2.0  
        return loss


class ConstrastiveLoss2(nn.Module):
    def __init__(self, init_tau=0.07):
        super().__init__()
        self.log_tau = nn.Parameter(torch.tensor(torch.log(torch.tensor(1.0 / init_tau))))

    def remove_row_col(self, T, j):
        B = T.shape[0]
        row_mask = torch.ones(B, dtype=torch.bool)
        col_mask = torch.ones(B, dtype=torch.bool)
        row_mask[j] = False
        col_mask[j] = False
        return T[row_mask][:, col_mask] 

    def forward(self, out_ftir, out_raman, labels):
        tau = torch.exp(self.log_tau).clamp(max=100)
        logits = (out_ftir @ out_raman.T) * tau
        
        labels = labels.view(-1, 1)
        positive_mask = (labels == labels.T).float()
        batch_size = positive_mask.shape[0]
        i_indices, j_indices = torch.where((positive_mask == 1.) & (torch.eye(batch_size, device=positive_mask.device) == 0))
        for i, j in zip(i_indices, j_indices):
            logits_neg = self.remove_row_col(logits, j)



        raman_loss = F.cross_entropy(logits, targets)
        ftir_loss = F.cross_entropy(logits.T, targets)
        loss = (ftir_loss + raman_loss) / 2.0  
        return loss

