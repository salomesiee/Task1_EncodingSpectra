import torch.nn.functional as F
import torch.nn as nn 
import torch 


class CLIPLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_tau = nn.Parameter(torch.zeros([]))

    def forward(self, out_ftir, out_raman):
        logits = (out_ftir @ out_raman.T) * torch.exp(self.log_tau)
        targets = torch.arange(len(logits), device=logits.device).long()

        raman_loss = F.cross_entropy(logits, targets)
        ftir_loss =  F.cross_entropy(logits.T, targets.T)
        loss = (ftir_loss + raman_loss) / 2.0  
        return loss
