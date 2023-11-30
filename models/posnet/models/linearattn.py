import torch
import torch.nn as nn

from . import register_model
from pytorchmt.util.timer import Timer
from pytorchmt.model.model import MTModel
from pytorchmt.util.globals import Globals
from pytorchmt.model.state import DynamicState
from pytorchmt.model.modules import (
    SinusodialPositionalEmbedding,
    PositionalEmbedding,
    LayerNormalization,
    MultiHeadAttention,
    Transpose,
)

@register_model("LinearAttn")
class LinearAttn(MTModel):

    def __init__(self, **kwargs):
        super().__init__()

        for k, v in kwargs.items():
            setattr(self, k, v)

        self.encoder = LinearAttnEncoder(**kwargs)
        self.decoder = LinearAttnDecoder(**kwargs)

        if self.tiew:
            self.decoder.tgt_embed.object_to_time.word_embed.weight = self.encoder.src_embed.object_to_time.word_embed.weight
            self.decoder.output_projection.object_to_time.weight = self.encoder.src_embed.object_to_time.word_embed.weight

    @staticmethod
    def create_model_from_config(config, vocab_src, vocab_tgt):
        
        model =  LinearAttn(
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
            linearattn_enc_self_att = config['linearattn_enc_self_att'],
            linearattn_dec_self_att = config['linearattn_dec_self_att'],
            linearattn_dec_cross_att = config['linearattn_dec_cross_att'],
            use_sinusodial_pos_embed = config['use_sinusodial_pos_embed', True],
            gating = config['gating', False]
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


class LinearAttnEncoder(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()

        for k, v in kwargs.items():
            setattr(self, k, v)

        if self.use_sinusodial_pos_embed:
            self.src_embed = Timer(SinusodialPositionalEmbedding(self.srcV, self.model_dim, self.maxI, self.dropout, self.pad_index))
        else:
            self.src_embed = Timer(PositionalEmbedding(self.srcV, self.model_dim, self.maxI, self.dropout, self.pad_index))
        
        self.layers = nn.ModuleList([LinearAttnEncoderLayer(**kwargs) for n in range(self.encL)])

        self.lnorm = Timer(LayerNormalization(self.model_dim))

    def __call__(self, src, src_mask=None, tgt_mask=None):

        h = self.src_embed(src)

        for layer in self.layers:

            h = layer(h, src_mask=src_mask)

        h = self.lnorm(h)

        return h


class LinearAttnDecoder(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()

        for k, v in kwargs.items():
            setattr(self, k, v)

        if self.use_sinusodial_pos_embed:
            self.tgt_embed = Timer(SinusodialPositionalEmbedding(self.tgtV, self.model_dim, self.maxI, self.dropout, self.pad_index))
        else:
            self.tgt_embed = Timer(PositionalEmbedding(self.tgtV, self.model_dim, self.maxI, self.dropout, self.pad_index))

        self.layers = nn.ModuleList([LinearAttnDecoderLayer(**kwargs) for n in range(self.decL)])

        self.lnorm              = Timer(LayerNormalization(self.model_dim))
        self.output_projection  = Timer(nn.Linear(self.model_dim, self.tgtV))
        self.log_softmax        = Timer(nn.LogSoftmax(dim=-1))

    def __call__(self, tgt, h, i=None, src_mask=None, tgt_mask=None):

        s = self.tgt_embed(tgt, J=i)

        for layer in self.layers:

            s = layer(s, h, src_mask=src_mask, tgt_mask=tgt_mask)

        s = self.lnorm(s)
        s = self.output_projection(s)
        s = self.log_softmax(s)

        return s


class LinearAttnEncoderLayer(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()

        for k, v in kwargs.items():
            setattr(self, k, v)
        
        self.lnorm1     = Timer(LayerNormalization(self.model_dim))
        if self.linearattn_enc_self_att:
            self.att    = LinearMultiHeadAttentionLayer(
                self.nHeads, self.model_dim, self.maxI, self.dropout, False,
                gating=self.gating
            )
        else:
            self.att    = MultiHeadAttention(self.nHeads, self.model_dim, self.dropout)
        self.dropout    = Timer(nn.Dropout(self.dropout))

        self.lnorm2     = Timer(LayerNormalization(self.model_dim))
        self.ff1        = Timer(nn.Linear(self.model_dim, self.ff_dim))
        self.relu       = Timer(nn.ReLU())
        self.ff2        = Timer(nn.Linear(self.ff_dim, self.model_dim))

    def __call__(self, x, src_mask=None):

        J = x.shape[1]
        
        r = x
        x = self.lnorm1(x)
        if self.linearattn_enc_self_att:
            x, _ = self.att(x, x, J, m=src_mask)
        else:
            x, _ = self.att(x, x, x, m=src_mask)
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


class LinearAttnDecoderLayer(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()

        for k, v in kwargs.items():
            setattr(self, k, v)

        self.lnorm1         = Timer(LayerNormalization(self.model_dim))
        if self.linearattn_dec_self_att:
            self.self_att   = LinearMultiHeadAttentionLayer(
                self.nHeads, self.model_dim, self.maxI, self.dropout, self.stepwise,
                gating=self.gating
            )
        else:
            self.self_att   = MultiHeadAttention(self.nHeads, self.model_dim, self.dropout)
        self.self_att_state = DynamicState(time_dim=1, stepwise=self.stepwise)

        self.lnorm2         = Timer(LayerNormalization(self.model_dim))
        if self.linearattn_dec_cross_att:
            self.cross_att  = LinearMultiHeadAttentionLayer(
                self.nHeads, self.model_dim, self.maxI, self.dropout, self.stepwise
            )
        else:
            self.cross_att  = MultiHeadAttention(self.nHeads, self.model_dim, self.dropout)

        self.lnorm3     = Timer(LayerNormalization(self.model_dim))
        self.ff1        = Timer(nn.Linear(self.model_dim, self.ff_dim))
        self.relu       = Timer(nn.ReLU())
        self.ff2        = Timer(nn.Linear(self.ff_dim, self.model_dim))
        self.dropout    = Timer(nn.Dropout(self.dropout))

    def __call__(self, s, h, src_mask=None, tgt_mask=None):
        
        r = s
        s = self.lnorm1(s)
        s_full = self.self_att_state.full(s)
        if self.linearattn_dec_self_att:
            s, _ = self.self_att(s_full, s, s_full.shape[1], m=tgt_mask)
        else:
            s, _ = self.self_att(s, s_full, s_full, m=tgt_mask)
        s = self.dropout(s)
        s = s + r

        r = s
        s = self.lnorm2(s)
        if self.linearattn_dec_cross_att:
            s, _ = self.cross_att(h, s_full.shape[1], m=src_mask)
        else:
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


class LinearMultiHeadAttentionLayer(nn.Module):

    def __init__(self, 
        H, D, maxI, dropout, stepwise,
        gating=False
    ):
        super().__init__()

        self.H = H
        self.D = D
        self.Dh = D // H

        self.gating = gating
        self.stepwise = stepwise
        
        self.W_v = nn.Linear(D, D)

        if gating:
            self.W_g = nn.Linear(D, D)
            self.act_v = nn.GELU()
            self.act_g = nn.GELU()
            self.lnorm_v = LayerNormalization(D)

        self.W = nn.Parameter(torch.ones(1, H, maxI, maxI), requires_grad=True)

        self.W_o = nn.Linear(D, D)

        self.transpose = Transpose()
        self.softmax = nn.Softmax(-1)
        self.dropout = nn.Dropout(dropout)

    def __call__(self, v, g, L_out, m=None):
        
        B = v.shape[0]
        I = v.shape[1]
        H = self.H
        Dh = self.Dh

        v = self.W_v(v)

        if self.gating:
            v = self.act_v(v)
            v = self.lnorm_v(v)

            g = self.W_g(g)
            g = self.act_g(g)

            g = g.view(B, -1, H, Dh)
            g = self.transpose(g, 1, 2)

        v = v.view(B, -1, self.H, self.Dh)
        v = self.transpose(v, 1, 2)

        if Globals.is_training() or not self.stepwise:
            a = self.W[:,:,:L_out,:I]
        else:
            a = self.W[:,:,L_out-1,:I]
            a = a.view(-1, H, 1, I)

        if m is not None:
            a = a.masked_fill(m, 0.)

        a = self.dropout(a)

        o = torch.matmul(a, v)

        if self.gating:
            o = o * g

        o = self.transpose(o, 1, 2)
        o = o.reshape(B, -1, self.D)
        o = self.W_o(o)

        return o, a