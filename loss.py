import torch.nn.functional as F
import torch.nn as nn 
import torch 


class CLIPLoss(nn.Module):
    def __init__(self, init_tau=0.07):
        super().__init__()
        self.log_tau = nn.Parameter(torch.tensor(torch.log(torch.tensor(1.0 / init_tau))))

    def forward(self, out_ftir, out_raman):
        tau = torch.exp(self.log_tau).clamp(max=100)

        logits = (out_ftir @ out_raman.T) * tau
        targets = torch.arange(len(logits), device=logits.device).long()

        raman_loss = F.cross_entropy(logits, targets)
        ftir_loss =  F.cross_entropy(logits.T, targets)
        loss = (ftir_loss + raman_loss) / 2.0  
        return loss
