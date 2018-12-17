import torch.nn as nn

from .trainer import BaseTrainer


class AutoTaggingTrainer(BaseTrainer):
    """"""
    def __init__(self, train_dataset, model, l2=1e-7,
                 learn_rate=0.001, batch_size=128, n_epochs=200,
                 valid_dataset=None, loss_every=100, save_every=10,
                 n_mels=20, is_gpu=False, out_root=None, name=None,
                 checkpoint=None):
        """"""
        super().__init__(
            train_dataset, model, l2, learn_rate, batch_size,
            n_epochs, valid_dataset, loss_every, save_every,
            is_gpu, out_root, name, checkpoint
        )
        self.loss = nn.BCELoss()
        print('GPU training:', self.is_gpu)

    def _parse_data(self, batch):
        """"""
        # fetch data
        X = batch['signal'][:, None]
        y = batch['target']
        if 'noisy_signal' in batch:
            X = batch['noisy_signal'][:, None]
        else:
            X = X
        if self.is_gpu:
            X, y = X.cuda(), y.cuda()

        return (X, y)

    def _forward(self, *data):
        """"""
        (X, y) = data  # signal
        y_hat = torch.sigmoid(self.model(X))
        l = self.loss(y_hat, y)
        return l