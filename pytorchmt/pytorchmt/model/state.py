import torch
import torch.nn as nn

from pytorchmt.util.globals import Globals


class StaticState:
    """
    Just saves the result of a call to a module.
    """

    def __init__(self, module):
        self.module = module

    def __call__(self, *args, **kwargs):
        
        cache = self.module(*args, **kwargs)

        if not isinstance(cache, (list, tuple)):
            cache = [cache]
        elif isinstance(cache, tuple):
            cache = list(cache)
        
        self.cache = cache
        self.cached = True

        return self.cache

    def read(self, mask_select=None):

        assert self.cached

        if mask_select is not None:

            res = []
            for c in self.cache:
                shape = c.shape[2:]
                res.append(torch.masked_select(c, mask_select).view(-1, *shape))
            
            return res
        else:
            return self.cache

    def repeat(self, repeats, dim=-1):

        assert self.cached

        for i, _ in enumerate(self.cache):

            if dim == -1:
                cur_dim = len(self.cache[i].shape)-1
            else:
                cur_dim = dim

            self.cache[i] = self.cache[i].unsqueeze(dim)
            self.cache[i] = self.cache[i].repeat([repeats if d == cur_dim else 1 for d in range(len(self.cache[i].shape))])


class DynamicState(nn.Module):
    """
    In search, it dynamically handles a state.
    """

    def __init__(self, time_dim=1, stepwise=True, dtype=torch.float32):
        super().__init__()

        self.stepwise = stepwise
        
        if not Globals.is_training() and stepwise:
            self.dtype = dtype
            self.time_dim = time_dim
            self.cache = None
            self.prev_mask = None

    def full(self, s):
        """
        Appends the current state s to the cache.
        Returns all saved states and the current state s.
        """
        if Globals.is_training() or not self.stepwise:
            return s
        else:
            
            if self.cache is None:
                self.cache = s
                return s

            s = torch.cat([self.cache, s], dim=self.time_dim)

            self.cache = s

            return s

    def repeat_interleave(self, *args):
        self.cache = torch.repeat_interleave(self.cache, *args)

    def reorder(self, order):
        """
        Reorder the cache given the indices by 'order'.
        The indices are  expected to have a single dimension which must match the one of cache: self.cache.shape[0] == order.shape[0]
        """

        assert order.shape[0] == self.cache.shape[0]
        assert len(order.shape) == 1

        ones = [1]
        for s in self.cache.shape[1:]:
            order = order.unsqueeze(-1).repeat(*ones, s)
            ones.append(1)
        
        self.cache = torch.gather(self.cache, 0, order)
    
    def reduce(self, mask, cummulate_masks=True):
        """
        In beam search the number of beams, and hence the batch dimension, is steadily decreased.
        Because the mask is not reduced in the same manner, the previous reduction must be applied 
        to it before it can be used to reduce the current state. This is done when setting
        cummulate_masks to True.
        """

        assert len(mask.shape) == 1
        
        if cummulate_masks and self.prev_mask is not None:
            m = torch.masked_select(mask, self.prev_mask)
        else:
            m = mask

        assert m.shape[0] == self.cache.shape[0]

        shape = self.cache.shape[1:]
        for _ in shape:
            m = m.unsqueeze(-1)

        self.cache = torch.masked_select(self.cache, m)
        self.cache = self.cache.view(-1, *shape)

        if cummulate_masks:
            self.prev_mask = mask

    def clear(self):
        self.cache = None
        self.prev_mask = None
