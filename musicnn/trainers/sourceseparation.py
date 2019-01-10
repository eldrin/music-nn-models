import torch
import torch.nn as nn

from .trainer import BaseTrainer


class SourceSeparationTrainer(BaseTrainer):
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
        # self.loss = nn.MSELoss()
        self.loss = nn.L1Loss()
        print('GPU training:', self.is_gpu)

    def _parse_data(self, batch):
        """"""
        # fetch data
        X = batch['mixture']
        Y = batch['vocals']
        if 'noisy_signal' in batch:
            X = batch['noisy_signal']
        else:
            X = X
        if self.is_gpu:
            X, Y = X.cuda(), Y.cuda()

        return (X, Y)

    def _forward(self, *data):
        """"""
        (X, Y) = data  # signal

        # # get acompaniment
        # Ya, _ = self.model._preproc(X - Y)

        # transform data
        # Yv = self.model._preproc(Y)
        Yv, _ = self.model._preproc(Y)
        Yv /= (Yv.max() + 1e-10)

        # prediction
        X, z = self.model._preproc(X)  # scaled / not scaled
        X /= (X.max() + 1e-10)  # to normalize it within [0, 1]

        # get mask
        M = self.model.get_mask(z)

        # # get estmations
        Yv_ = M * X
        # Yv_, Ya_ = M * X, (1 - M) * X
        # Y_ = Yv_ + Ya_

        # calc loss
        l = self.loss(Yv_, Yv)
        # l = self.loss(Yv_, Yv) + self.loss(Ya_, Ya)

        return l
