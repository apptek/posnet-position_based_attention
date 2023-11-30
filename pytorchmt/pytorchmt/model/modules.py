import math

import torch
import torch.nn as nn

from pytorchmt.util.timer import Timer
from pytorchmt.util.globals import Globals


class MultiHeadAttention(nn.Module):

    def __init__(self, H, D, dropout):
        super().__init__()

        self.H = H
        self.D = D
        self.Dh = D // H

        self.att = DotProductAttention(dropout)
        
        self.W_q = Timer(nn.Linear(D, D))
        self.W_k = Timer(nn.Linear(D, D))
        self.W_v = Timer(nn.Linear(D, D))
        self.W_o = Timer(nn.Linear(D, D))

        self.transpose = Timer(Transpose())

    def __call__(self, q, k, v, m=None):
        
        B = q.shape[0]

        q = self.W_q(q)
        k = self.W_k(k)
        v = self.W_v(v)

        q = q.view(B, -1, self.H, self.Dh)
        k = k.view(B, -1, self.H, self.Dh)
        v = v.view(B, -1, self.H, self.Dh)

        q = self.transpose(q, 1, 2)
        k = self.transpose(k, 1, 2)
        v = self.transpose(v, 1, 2)

        o, a = self.att(q, k, v, m)

        o = self.transpose(o, 1, 2)
        o = o.reshape(B, -1, self.D)
        o = self.W_o(o)

        return o, a


class DotProductAttention(nn.Module):

    def __init__(self, dropout):
        super().__init__()

        self.matmul = Timer(MatMul())
        self.transpose = Timer(Transpose())
        self.softmax = Timer(nn.Softmax(-1))
        self.dropout = Timer(nn.Dropout(dropout))

    def __call__(self, q, k, v, m):
        
        D = q.shape[-1]

        k = self.transpose(k, -2, -1)

        a = self.matmul(q, k)
        a = a / math.sqrt(D)

        if m is not None:
            a = a.masked_fill(m, -float('inf'))

        a = self.softmax(a)
        a = self.dropout(a)

        o = self.matmul(a, v)

        return o, a


class PositionalEmbedding(nn.Module):

    def __init__(self, V, model_dim, maxI, dropout, pad_index, use_pos_embed=True, return_pos_embed=False):
        super().__init__()

        self.model_dim = model_dim
        self.return_pos_embed = return_pos_embed
        self.use_pos_embed = use_pos_embed

        self.word_embed = nn.Embedding(V, model_dim, padding_idx=pad_index)
        if self.use_pos_embed or self.return_pos_embed:
            self.pos_embed = nn.Embedding(maxI, model_dim)
        self.dropout = nn.Dropout(dropout)

        rng = torch.arange(maxI)
        self.register_buffer('rng', rng)

    def __call__(self, x, J=None):

        B = x.shape[0]
        D = self.model_dim

        x = self.word_embed(x)

        if self.use_pos_embed or self.return_pos_embed:
            
            if J is None:
                J = x.shape[1]
                pos = self.pos_embed(self.rng[:J]) 
            else:
                assert x.shape[1] == 1
                pos = self.pos_embed(self.rng[J-1])

            pos = pos.unsqueeze(0).repeat(B, 1, 1)

            if self.use_pos_embed:
                x = x + pos
        
        x = x * math.sqrt(D)

        x = self.dropout(x)

        if self.return_pos_embed:
            return x, pos
        else:
            return x


class SinusodialPositionalEmbedding(nn.Module):

    def __init__(self, V, D, I, dropout, pad_index, use_pos_embed=True, return_pos_embed=False):
        super().__init__()

        self.D = D
        self.return_pos_embed = return_pos_embed
        self.use_pos_embed = use_pos_embed

        self.word_embed = nn.Embedding(V, D, padding_idx=pad_index)
        self.dropout = nn.Dropout(dropout)

        self.__precalculate_pos_embed(D, I)

    def __precalculate_pos_embed(self, D, I):

        import math

        pos_embed = torch.arange(I, dtype=torch.float).unsqueeze(-1).repeat(1, D)
        pos = torch.arange(I, dtype=torch.float).unsqueeze(-1)
        i = torch.arange(0, D, 2, dtype=torch.float)

        i = torch.exp(-(i / D) * math.log(10000))
        pos = pos * i

        pos_embed[:, 0::2] = torch.sin(pos)
        pos_embed[:, 1::2] = torch.cos(pos)

        self.register_buffer('pos_embed', pos_embed.to(Globals.get_device()))

    def __call__(self, x, J=None):

        B = x.shape[0]
        D = self.D

        x = self.word_embed(x)

        x = x * math.sqrt(D)

        if self.use_pos_embed or self.return_pos_embed:

            if J is None:
                J = x.shape[1]
                pos = self.pos_embed[:J]
            else:
                assert x.shape[1] == 1
                pos = self.pos_embed[J-1]

            pos = pos.unsqueeze(0).repeat(B, 1, 1)  

            if self.use_pos_embed:
                x = x + pos

        x = self.dropout(x)

        if self.return_pos_embed:
            return x, pos
        else:
            return x


class LayerNormalization(nn.Module):

    def __init__(self, model_dim):
        super().__init__()

        self.a = nn.Parameter(torch.ones(model_dim))
        self.b = nn.Parameter(torch.zeros(model_dim))

    def __call__(self, x):
        
        mu = torch.mean(x, dim=-1, keepdim=True)
        sg = torch.var(x, dim=-1, keepdim=True)

        x = (x - mu) / torch.sqrt(sg + 1e-8)
        x = x * self.a + self.b

        return x


class Transpose(nn.Module):

    def __init__(self):
        super().__init__()

    def __call__(self, *args, **kwargs):
        return torch.transpose(*args, **kwargs)


class MatMul(nn.Module):

    def __init__(self):
        super().__init__()

    def __call__(self, *args, **kwargs):
        return torch.matmul(*args, **kwargs)