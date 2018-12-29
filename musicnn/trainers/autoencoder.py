import torch
import torch.nn as nn
import torch.nn.functional as F

from .trainer import BaseTrainer


class AutoEncoderTrainer(BaseTrainer):
    """"""
    def __init__(self, train_dataset, model, l2=1e-7,
                 learn_rate=0.001, batch_size=128, n_epochs=200,
                 valid_dataset=None, loss_every=100, save_every=10,
                 is_gpu=False, out_root=None, name=None, n_jobs=4,
                 checkpoint=None, n_valid_batches=1):
        """"""
        super().__init__(
            train_dataset, model, l2, learn_rate, batch_size,
            n_epochs, valid_dataset, loss_every, save_every,
            is_gpu, out_root, name, n_jobs, checkpoint, n_valid_batches
        )
        self.loss = nn.MSELoss()
        # self.loss = kl_with_logits

        print('GPU training:', self.is_gpu)

    def _parse_data(self, batch):
        """"""
        # fetch data
        X = batch['signal']
        if 'noisy_signal' in batch:
            X = batch['noisy_signal']
        else:
            X = X
        if self.is_gpu:
            X = X.cuda()

        return (X,)

    def _forward(self, *data):
        """"""
        (X,) = data  # signal
        S, Shat = self.model(X)  # spectrogram / prediction of it
        l = self.loss(Shat, S)
        return l


def kl_with_logits(x, y, dim=2, eps=1e-10):
    """Helper function for getting KL with logits as input

    Args:
        x (torch.tensor): input prediction
        y (torch.tensor): target prob. distribution
        dim (int): dimension where the variables spanned over
        eps (float): small number for preventing overflow
    """
    eps = torch.tensor([eps])
    if x.is_cuda:
        eps = eps.cuda()
        
    return F.kl_div(
        torch.max(F.softmax(x, dim=dim), eps).log(),
        y, reduction='sum'
    ) / x.shape[0]
