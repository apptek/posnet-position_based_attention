
import math

import torch
import torch.nn as nn

from . import register_model
from pytorchmt.util.timer import Timer
from pytorchmt.util.debug import my_print
from pytorchmt.model.model import MTModel
from pytorchmt.util.globals import Globals
from pytorchmt.model.state import DynamicState
from pytorchmt.model.modules import (
    SinusodialPositionalEmbedding,
    PositionalEmbedding,
    MultiHeadAttention,
    LayerNormalization,
    Transpose,
    MatMul
)

@register_model("RelTransformer")
class RelTransformer(MTModel):

    def __init__(self, **kwargs):
        super().__init__()

        for k, v in kwargs.items():
            setattr(self, k, v)

        self.encoder = RelTransformerEncoder(**kwargs)
        self.decoder = RelTransformerDecoder(**kwargs)

        if self.tiew:
            self.decoder.tgt_embed.object_to_time.word_embed.weight = self.encoder.src_embed.object_to_time.word_embed.weight
            self.decoder.output_projection.object_to_time.weight = self.encoder.src_embed.object_to_time.word_embed.weight

    @staticmethod
    def create_model_from_config(config, vocab_src, vocab_tgt):
        
        model =  RelTransformer(
            pad_index = vocab_src.PAD,
            srcV = vocab_src.vocab_size,
            tgtV = vocab_tgt.vocab_size,
            encL = config['encL'],
            decL = config['decL'],
            model_dim = config['model_dim'],
            nHeads = config['nHeads'],
            ff_dim = config['ff_dim'],
            dropout = config['dropout'],
            maxI = config['max_sentence_length'],
            tiew = config['tiew'],
            initializer = config['initializer'],
            variance_scaling_scale = config['variance_scaling_scale'],
            stepwise = config['stepwise'],
            K = config['K'],
            use_pos_embed = config['use_pos_embed', False],
            use_sinusodial_pos_embed = config['use_sinusodial_pos_embed', False],
            gating=config['gating', False],
            only_rel_attention=config['only_rel_attention', False]
        )

        return model

    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def __call__(self, src, tgt, src_mask=None, tgt_mask=None):
        
        h = self.encoder(src, src_mask=src_mask, tgt_mask=tgt_mask)

        s = self.decoder(tgt, h, src_mask=src_mask, tgt_mask=tgt_mask)

        return s, h

    def create_masks(self, src, tgt, pad_index):

        masks = {}

        src_mask = (src == pad_index)
        src_mask = src_mask.unsqueeze(1).unsqueeze(2)
        masks['src_mask'] = src_mask

        if tgt is not None:

            tgtT = tgt.shape[1]

            tgt_mask = torch.tril(tgt.new_ones((tgtT, tgtT)))
            tgt_mask = tgt_mask.unsqueeze(0).unsqueeze(1)  # B, H, I, I
            tgt_mask = (tgt_mask == 0)

            masks['tgt_mask'] = tgt_mask

        out_mask = (tgt != pad_index)

        return masks, out_mask


class RelTransformerEncoder(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()

        for k, v in kwargs.items():
            setattr(self, k, v)

        if self.use_sinusodial_pos_embed:
            self.src_embed = Timer(SinusodialPositionalEmbedding(self.srcV, self.model_dim, self.maxI, self.dropout, self.pad_index, use_pos_embed=self.use_pos_embed))
        else:
            self.src_embed = Timer(PositionalEmbedding(self.srcV, self.model_dim, self.maxI, self.dropout, self.pad_index, use_pos_embed=self.use_pos_embed))
        
        self.layers = nn.ModuleList([RelTransformerEncoderLayer(**kwargs) for n in range(self.encL)])

        self.lnorm = Timer(LayerNormalization(self.model_dim))

    def __call__(self, src, src_mask=None, tgt_mask=None):

        h = self.src_embed(src)

        for layer in self.layers:

            h = layer(h, src_mask=src_mask)

        h = self.lnorm(h)

        return h


class RelTransformerDecoder(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()

        for k, v in kwargs.items():
            setattr(self, k, v)

        if self.use_sinusodial_pos_embed:
            self.tgt_embed = Timer(SinusodialPositionalEmbedding(self.tgtV, self.model_dim, self.maxI, self.dropout, self.pad_index, use_pos_embed=self.use_pos_embed))
        else:
            self.tgt_embed = Timer(PositionalEmbedding(self.tgtV, self.model_dim, self.maxI, self.dropout, self.pad_index, use_pos_embed=self.use_pos_embed))

        self.layers = nn.ModuleList([RelTransformerDecoderLayer(**kwargs) for n in range(self.decL)])

        self.lnorm              = Timer(LayerNormalization(self.model_dim))
        self.output_projection  = Timer(nn.Linear(self.model_dim, self.tgtV))
        self.log_softmax        = Timer(nn.LogSoftmax(dim=-1))

    def __call__(self, tgt, h, i=None, src_mask=None, tgt_mask=None):

        s = self.tgt_embed(tgt, J=i)

        for layer in self.layers:

            s = layer(s, h, i=i, src_mask=src_mask, tgt_mask=tgt_mask)

        s = self.lnorm(s)
        s = self.output_projection(s)
        s = self.log_softmax(s)

        return s


class RelTransformerEncoderLayer(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()

        for k, v in kwargs.items():
            setattr(self, k, v)
        
        self.lnorm1     = Timer(LayerNormalization(self.model_dim))
        self.att        = RelativeMultiHeadAttention(
            self.nHeads, self.model_dim, self.K, self.maxI, self.dropout,
            only_rel_attention=self.only_rel_attention,
            gating=self.gating
        )
        self.dropout    = Timer(nn.Dropout(self.dropout))

        self.lnorm2     = Timer(LayerNormalization(self.model_dim))
        self.ff1        = Timer(nn.Linear(self.model_dim, self.ff_dim))
        self.relu       = Timer(nn.ReLU())
        self.ff2        = Timer(nn.Linear(self.ff_dim, self.model_dim))

    def __call__(self, x, src_mask=None):
        
        r = x
        x = self.lnorm1(x)
        x, _ = self.att(x, x, x, x, m=src_mask)
        x = self.dropout(x)
        x = x + r

        r = x
        x = self.lnorm2(x)
        x = self.ff1(x)
        x = self.relu(x)
        x = self.ff2(x)
        x = self.dropout(x)
        x = x + r

        return x


class RelTransformerDecoderLayer(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()

        for k, v in kwargs.items():
            setattr(self, k, v)

        self.lnorm1         = Timer(LayerNormalization(self.model_dim))
        self.self_att       = RelativeMultiHeadAttention(
            self.nHeads, self.model_dim, self.K, self.maxI, self.dropout,
            only_rel_attention=self.only_rel_attention,
            stepwise=self.stepwise,
            gating=self.gating
        )
        self.self_att_state = DynamicState(time_dim=1, stepwise=self.stepwise)

        self.lnorm2     = Timer(LayerNormalization(self.model_dim))
        self.cross_att  = MultiHeadAttention(self.nHeads, self.model_dim, self.dropout)

        self.lnorm3     = Timer(LayerNormalization(self.model_dim))
        self.ff1        = Timer(nn.Linear(self.model_dim, self.ff_dim))
        self.relu       = Timer(nn.ReLU())
        self.ff2        = Timer(nn.Linear(self.ff_dim, self.model_dim))
        self.dropout    = Timer(nn.Dropout(self.dropout))

    def __call__(self, s, h, i=None, src_mask=None, tgt_mask=None):

        r = s
        s = self.lnorm1(s)
        s_full = self.self_att_state.full(s)
        s, _ = self.self_att(s, s_full, s_full, s, i_scalar=i, m=tgt_mask)
        s = self.dropout(s)
        s = s + r

        r = s
        s = self.lnorm2(s)
        s, _ = self.cross_att(s, h, h, m=src_mask)
        s = self.dropout(s)
        s = s + r

        r = s
        s = self.lnorm3(s)
        s = self.ff1(s)
        s = self.relu(s)
        s = self.ff2(s)
        s = self.dropout(s)
        s = s + r

        return s


class RelativeMultiHeadAttention(nn.Module):

    def __init__(self,
        H, D, K, maxI, dropout,
        only_rel_attention=False,
        stepwise=False,
        gating=False
    ):
        super().__init__()

        self.H = H
        self.D = D
        self.Dh = D // H
        self.K = K

        self.gating = gating
        self.stepwise = stepwise
        self.only_rel_attention = only_rel_attention

        self.__create_learnable_parameters(D, K, gating)
        self.__create_normalizations(D, gating)
        self.__create_activations(gating)
        self.__precalculate_indices(K, H, maxI)

        self.transpose = Timer(Transpose())
        self.softmax = Timer(nn.Softmax(-1))
        self.dropout = Timer(nn.Dropout(dropout))
        self.matmul = Timer(MatMul())

    def __create_learnable_parameters(self, D, K, gating):

        self.W_q = Timer(nn.Linear(D, D))
        self.W_o = Timer(nn.Linear(D, D))
        self.W_v = Timer(nn.Linear(D, D))

        if not self.only_rel_attention:
            self.W_k = Timer(nn.Linear(D, D))

        self.rel_embed = nn.Embedding(2*K+1, D)
        self.register_buffer('rng', torch.arange(2*K+1))

        if gating:
            self.W_g = Timer(nn.Linear(D, D))
        else:
            self.W_g = nn.Identity()
    
    def __create_normalizations(self, D, gating):
        if gating:
            self.lnorm_v = Timer(LayerNormalization(self.D))
        else:
            self.lnorm_v = nn.Identity()

    def __create_activations(self, gating):
        if gating:
            self.act_v = nn.GELU()
            self.act_g = nn.GELU()
        else:
            self.act_v = nn.Identity()
            self.act_g = nn.Identity()

    def __precalculate_indices(self, K, H, maxI):

        arangeI = torch.arange(maxI).to(Globals.get_device())
        indices = arangeI.unsqueeze(0)
        indices = indices.repeat(maxI, 1)
        indices = indices - arangeI.unsqueeze(1)
        indices = torch.clamp(indices, max=K, min=-K)
        indices += K
        indices = indices.unsqueeze(0).unsqueeze(0)
        indices = indices.repeat(1, H, 1, 1)

        self.register_buffer('indices', indices)

    def __call__(self, q, k, v, g, i_scalar=None, m=None):
        
        B = v.shape[0]
        I = v.shape[1]
        H = self.H
        D = self.D
        K = self.K
        Dh = self.Dh

        q = self.W_q(q)
        r = self.rel_embed(self.rng) # [2K+1, D]

        v = self.W_v(v)
        g = self.W_g(g)

        v = self.act_v(v)
        g = self.act_g(g)

        v = self.lnorm_v(v)

        q = q.view(B, -1, H, Dh)
        v = v.view(B, -1, H, Dh)
        r = r.view(1,  2*K+1, H, Dh)
        
        q = self.transpose(q, 1, 2)
        v = self.transpose(v, 1, 2)
        r = self.transpose(r, 1, 2)

        if self.gating:
            g = g.view(B, -1, H, Dh)
            g = self.transpose(g, 1, 2)

        a = self.matmul(q, self.transpose(r, -2, -1))

        if not Globals.is_training() and self.stepwise:
                assert i_scalar is not None
                indices = self.indices[:,:,i_scalar-1,:I].unsqueeze(-2)
        else:
            indices = self.indices[:,:,:I,:I]

        indices = indices.repeat(B, 1, 1, 1)
        a = torch.gather(a, -1, indices)

        if not Globals.is_training() and self.stepwise:
            assert list(a.shape) == [B, H, 1, I]
        else:
            assert list(a.shape) == [B, H, I, I]  

        if not self.only_rel_attention:
            k = self.W_k(k)
            k = k.view(B, -1, H, Dh)
            k = self.transpose(k, 1, 2)
            a = a + self.matmul(q, self.transpose(k, -2, -1))

        a = a / math.sqrt(Dh)

        if m is not None:
            a = a.masked_fill(m, -float('inf'))

        a = self.softmax(a)
        a = self.dropout(a)

        o = self.matmul(a, v)
        
        if self.gating:
            o = o * g

        o = self.transpose(o, 1, 2)
        o = o.reshape(B, -1, D)
        o = self.W_o(o)

        return o, a