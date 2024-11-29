import torch
from torchmetrics import Metric


class d1(Metric):
    full_state_update = True
    higher_is_better = True

    def __init__(self):
        super().__init__()
        self.add_state('under_d1', default=torch.tensor(0), dist_reduce_fx='sum')
        self.add_state('total_d1', default=torch.tensor(0), dist_reduce_fx='sum')

    def update(self, preds, target):
        x = torch.max((target / preds), (preds / target))
        self.under_d1 += torch.sum(x < 1.25)
        self.total_d1 += target.numel()

    def compute(self):
        return self.under_d1.float() / self.total_d1


class d2(Metric):
    full_state_update = True
    higher_is_better = True

    def __init__(self):
        super().__init__()
        self.add_state('under_d2', default=torch.tensor(0), dist_reduce_fx='sum')
        self.add_state('total_d2', default=torch.tensor(0), dist_reduce_fx='sum')

    def update(self, preds, target):
        x = torch.max((target / preds), (preds / target))
        self.under_d2 += torch.sum(x < 1.25**2)
        self.total_d2 += target.numel()

    def compute(self):
        return self.under_d2.float() / self.total_d2


class d3(Metric):
    full_state_update = True
    higher_is_better = True

    def __init__(self):
        super().__init__()
        self.add_state('under_d3', default=torch.tensor(0), dist_reduce_fx='sum')
        self.add_state('total_d3', default=torch.tensor(0), dist_reduce_fx='sum')

    def update(self, preds, target):
        x = torch.max((target / preds), (preds / target))
        self.under_d3 += torch.sum(x < 1.25**3)
        self.total_d3 += target.numel()

    def compute(self):
        return self.under_d3.float() / self.total_d3
