import os
from os.path import join, exists, basename, dirname
import time
from functools import partial
from itertools import chain
import numpy as np

import librosa

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader  # dataset fecther related

from tensorboardX import SummaryWriter  # tensorboard logger

from tqdm import tqdm, trange

from ..config import Config as cfg
from ..utils.utils import save_checkpoint, load_checkpoint


class BaseTrainer(object):
    """ Base trainer class """
    def __init__(self, train_dataset, model, l2=1e-7,
                 learn_rate=0.001, batch_size=128, n_epochs=200,
                 valid_dataset=None, loss_every=100, save_every=10,
                 is_gpu=False, out_root=None, name=None, n_jobs=4,
                 checkpoint=None, n_valid_batches=1):
        """"""
        self.train_dataset = DataLoader(train_dataset,
                                        batch_size=batch_size,
                                        shuffle=True,
                                        num_workers=n_jobs)
        if valid_dataset is not None:
            self.valid_dataset = DataLoader(valid_dataset,
                                            batch_size=batch_size,
                                            shuffle=True,
                                            num_workers=n_jobs)
        self.model = model
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.learn_rate = learn_rate
        self.l2 = l2
        self.is_gpu = is_gpu
        self.loss_every = loss_every  # frequency to write train loss to logger
        self.save_every = save_every  # frequency to save the model
        self.n_valid_batches = n_valid_batches

        if len(list(filter(lambda module: hasattr(module, 'sparse'),
                           self.model.children()))) > 0:
            # filter out 
            sprs_prms = chain.from_iterable([
                module.parameters() for module
                in filter(
                    lambda module: hasattr(module, 'sparse'),
                    self.model.children()
                )
            ])
            dnse_prms = chain.from_iterable([
                module.parameters() for module
                in filter(
                    lambda module: not hasattr(module, 'sparse'),
                    self.model.children()
                )
            ])

            # setup multi optimizer
            self.opt = MultipleOptimizer(
                optim.SparseAdam(sprs_prms, lr=self.learn_rate),
                optim.Adam(dnse_prms, lr=self.learn_rate)
            )
        else:
            self.opt = optim.Adam(
                filter(lambda w: w.requires_grad, self.model.parameters()),
                lr = self.learn_rate,
                weight_decay = self.l2,
                amsgrad=True
            )
        # self.opt = optim.SGD(
        #     filter(lambda w: w.requires_grad, self.model.parameters()),
        #     lr = self.learn_rate,
        #     weight_decay = self.l2,
        #     momentum = 0.9,
        # )
        # self.lr_scheduler = optim.lr_scheduler.StepLR(
        #     self.opt, step_size=250, gamma=0.1)

        # multi-gpu
        if torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)
            self.multi_gpu = True
            print('[Info] Using {:d} gpus'.format(torch.cuda.device_count()))
        else:
            self.multi_gpu = False

        if is_gpu:
            self.model.cuda()

        # setting up the logging
        if name is None:
            if checkpoint is None:
                self.logger = SummaryWriter()
                self.name = ''
            else:
                self.name = basename(checkpoint).split('.')[0]
                self.logger = SummaryWriter('runs/{}'.format(self.name))
        else:
            self.logger = SummaryWriter('runs/{}'.format(name))
            self.name = name

        if out_root is None:
            self.out_root = os.getcwd()
        else:
            self.out_root = out_root

        # self.run_out_root = join(self.out_root, self.name)
        # if not os.path.exists(self.run_out_root):
        #     os.mkdir(self.run_out_root)

        self.iters = 0
        self.checkpoint = checkpoint
        # resuming training
        if checkpoint is not None and exists(checkpoint):
            cp = load_checkpoint(checkpoint)
            self.model.eval()
            self.model.load_state_dict(cp['state_dict'])
            self.opt.load_state_dict(cp['optimizer'])
            self.iters = cp['iters']

    def fit(self):
        """"""
        try:
            for n in trange(self.n_epochs, ncols=80):
                # self.lr_scheduler.step()

                # evaluation (only calc evaluation loss at the moment)
                if self.valid_dataset is not None:
                    self.model.eval()
                    counter = 0
                    vloss = 0
                    for j, batch in enumerate(self.valid_dataset):
                        vloss += self.partial_eval(batch)
                        counter += 1
                        if counter > self.n_valid_batches:
                            break  # only one batch for efficiency 
                    self.logger.add_scalar('vloss', vloss.item() / counter, self.iters)

                # training
                self.model.train()
                for i, batch in enumerate(self.train_dataset):
                    tloss = self.partial_fit(batch)
                    if self.iters % self.loss_every == 0:
                        # training log
                        self.logger.add_scalar('tloss', tloss.item(), self.iters)
                    self.iters += 1

                if self.save_every and (n % self.save_every == 0):
                    self.save(suffix='it{:d}'.format(n))

        except KeyboardInterrupt:
            print('[Warning] User stopped the training!')

        finally:
            # for the case where the training accidentally break
            self.model.eval()
            self.save()

    def partial_fit(self, batch):
        """"""
        # prepare update
        self.opt.zero_grad()

        # parsing data
        data = self._parse_data(batch)

        # forward
        l = self._forward(*data)

        # backward
        l.backward()

        # update params
        self.opt.step()
        return l

    def partial_eval(self, batch):
        """"""
        return self._forward(*self._parse_data(batch))

    def _parse_data(self, batch):
        """"""
        raise NotImplementedError()

    def _forward(self, *data):
        """"""
        raise NotImplementedError()

    def save(self, suffix=None):
        """"""
        if self.checkpoint is None:
            if self.name == '':
                self.name = 'model'
            self.checkpoint = join(os.getcwd(), self.name)

        if suffix:
            out_fn = self.checkpoint + '_{}.pth'.format(suffix)
        else:
            out_fn = self.checkpoint + '.pth'

        self.model.eval()
        state = {
            'iters': self.iters,
            'state_dict': self.model.state_dict(),
            'optimizer': self.opt.state_dict()
        }
        torch.save(state, out_fn)
