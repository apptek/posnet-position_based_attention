
from os import mkdir
from os.path import join, isdir

import torch
import torch.nn as nn
import horovod.torch as hvd

from pytorchmt.util.debug import my_print
from pytorchmt.util.globals import Globals


class Optimizer(nn.Module):

    def __init__(self, lr, numbers_dir):
        super().__init__()

        lr = torch.tensor(lr).to(torch.float32)
        self.register_buffer('lr', lr)

        self.numbers_dir = numbers_dir

    @staticmethod
    def create_optimizer_from_config(config, numbers_dir, model_params, named_parameters):

        if config['optimizer'] == 'WarmupAdam':
            return WarmupScheduledAdamOptimizer.create_optimizer_from_config(config, numbers_dir, model_params, named_parameters)

        else:
            assert True == False, 'Unknown score "%s"' % (config['score'])

    def write_lr_to_file(self, L):

        if hvd.rank() != 0:
            return

        if not isdir(self.numbers_dir):
            mkdir(self.numbers_dir)

        with open(join(self.numbers_dir, 'lr'), 'a') as file:
            file.write(f'lr {self.lr}, L {L}\n')

    def step(self, grads, vars):
        raise NotImplementedError()

    def update_lr(self):
        raise NotImplementedError()


class WarmupScheduledAdamOptimizer(Optimizer):

    def __init__(self, params, named_parameters, numbers_dir, **kwargs):
        super().__init__(0., numbers_dir)

        for k, v in kwargs.items():
            setattr(self, k, v)

        self.adam = hvd.DistributedOptimizer(
            optimizer = torch.optim.Adam(
                params,
                lr = self.lr,
                betas=(0.9, 0.98),
                eps=1e-9
            ),
            named_parameters = named_parameters
        )

        _step = torch.tensor(0.).to(torch.float32)
        self.register_buffer('_step', _step)

    @staticmethod
    def create_optimizer_from_config(config, numbers_dir, model_params, named_parameters):
        return WarmupScheduledAdamOptimizer(
            model_params,
            named_parameters,
            numbers_dir,
            model_dim = config["model_dim"], 
            warmup = config["warmup"],
            update_freq = config["update_freq"],
            lr_scale = config["lr_scale"],
        )

    def step(self):

        self._step += 1

        self.update_lr()

        self.adam.step()

        self.adam.zero_grad()

        return self.lr

    def update_lr(self):

        s  = self._step
        w  = self.warmup
        D  = self.model_dim
        e1 = -0.5
        e2 = -1.5

        self.lr = (D ** e1) * torch.minimum(s ** e1, s * w ** e2) * self.lr_scale

        for p in self.adam.param_groups:
            p['lr'] = self.lr