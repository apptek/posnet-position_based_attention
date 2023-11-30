
from math import log
from os import mkdir
from os.path import join, isdir

import torch
import torch.nn as nn
import horovod.torch as hvd


class Score(nn.Module):

    def __init__(self):
        super().__init__()

    @staticmethod
    def create_score_from_config(config):

        if config['score'] == 'LabelSmoothingCrossEntropy':
            
            return LabelSmoothingCrossEntropyLoss.create_score_from_config(config)

        else:

            assert True == False, 'Unknown score "%s"' % (config['score'])

    @staticmethod
    def write_score_to_file(directory, filename, score):

        if hvd.rank() != 0:
            return

        if not isdir(directory):
            mkdir(directory)

        if isinstance(score, torch.Tensor):
            score = float(score.cpu().detach().numpy())

        with open(join(directory, filename), 'a') as file:
            file.write(f'{score}\n')


class LabelSmoothingCrossEntropyLoss(Score):

    def __init__(self, m):
        super(LabelSmoothingCrossEntropyLoss, self).__init__()

        self.m = m
        
        self.ce = 0.
        self.ce_smooth = 0.
        self.L = 0.

    @staticmethod
    def create_score_from_config(config):

        return LabelSmoothingCrossEntropyLoss(
            config['label_smoothing']
        )

    def __call__(self, output, out, out_mask=None):

        tgtV = output.shape[-1]

        out         = out.reshape(-1, 1)
        output      = output.reshape(-1, tgtV)
        out_mask    = out_mask.reshape(-1, 1)
        
        m = self.m
        w = m / (tgtV - 1)

        nll_loss = -1 * output.gather(dim=-1, index=out)
        smo_loss = -1 * output.sum(-1, keepdim=True)

        nll_loss = nll_loss * out_mask
        smo_loss = smo_loss * out_mask
        
        ce_smooth = (1 - m - w) * nll_loss + w * smo_loss

        num_words = out_mask.sum()
        ce_smooth = ce_smooth.sum()
        ce = nll_loss.sum()

        self.ce += ce
        self.ce_smooth += ce_smooth
        self.L += num_words

        return ce, ce_smooth, num_words

    def average_and_reset(self):

        ce = hvd.allreduce(self.ce, average=False)
        ce_smooth = hvd.allreduce(self.ce_smooth, average=False)
        L = hvd.allreduce(self.L, average=False)

        ce = float(ce.cpu().detach().numpy())
        ce_smooth = float(ce_smooth.cpu().detach().numpy())
        L = float(L.cpu().detach().numpy())

        self.ce = 0.
        self.ce_smooth = 0.
        self.L = 0.

        return (ce / L), (ce_smooth / L)