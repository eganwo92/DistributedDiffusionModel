import copy
import torch

class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.ema_model = copy.deepcopy(model).eval()
        for p in self.ema_model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for p_ema, p in zip(self.ema_model.parameters(), model.parameters()):
            p_ema.data.mul_(self.decay).add_(p.data, alpha=1.0 - self.decay)
