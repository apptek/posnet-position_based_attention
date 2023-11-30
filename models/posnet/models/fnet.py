import math

import torch
import torch.nn as nn

from . import register_model
from pytorchmt.util.timer import Timer
from pytorchmt.model.model import MTModel
from pytorchmt.util.debug import my_print
from pytorchmt.util.globals import Globals
from pytorchmt.model.state import DynamicState
from pytorchmt.model.modules import (
    SinusodialPositionalEmbedding,
    LayerNormalization,
    MultiHeadAttention,
    Transpose,
)

@register_model("FNet")
class FNet(MTModel):

    def __init__(self, **kwargs):
        super().__init__()

        for k, v in kwargs.items():
            setattr(self, k, v)

        self.encoder = FNetEncoder(**kwargs)
        self.decoder = FNetDecoder(**kwargs)

        if self.tiew:
            self.decoder.tgt_embed.object_to_time.word_embed.weight = self.encoder.src_embed.object_to_time.word_embed.weight
            self.decoder.output_projection.object_to_time.weight = self.encoder.src_embed.object_to_time.word_embed.weight

    @staticmethod
    def create_model_from_config(config, vocab_src, vocab_tgt):
        
        model =  FNet(
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
            fnet_enc_self_att = config['fnet_enc_self_att'],
            fnet_dec_self_att = config['fnet_dec_self_att'],
            sentence_length_factor = config['sentence_length_factor'],
            just_like_fnet = config['just_like_fnet']
        )

        return model

    @staticmethod
    def calculate_dft_matrix(B, I, I_act):

        assert list(I_act.shape) == [B], f'{I_act.shape}, {B}'

        I_act = I_act.unsqueeze(1).unsqueeze(1)

        W_a = (torch.arange(I).unsqueeze(-1) * torch.arange(I).unsqueeze(0)).to(Globals.get_device())

        W_a = W_a.unsqueeze(0).repeat(B, 1, 1) / I_act

        I_act = 1 / torch.sqrt(I_act)

        W_dft = torch.polar(I_act, -2 * math.pi * W_a)

        assert list(W_dft.shape) == [B, I, I], f'{W_dft.shape}, {B}, {I}'

        return W_dft

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
        src_mask = src_mask.unsqueeze(1).unsqueeze(2) # [B, srcT, 1, 1]
        masks['src_mask'] = src_mask

        if tgt is not None:

            tgtT = tgt.shape[1]

            tgt_mask = torch.tril(tgt.new_ones((tgtT, tgtT)))
            tgt_mask = tgt_mask.unsqueeze(0).unsqueeze(1)  # B, H, I, I
            tgt_mask = (tgt_mask == 0)

            masks['tgt_mask'] = tgt_mask

        out_mask = (tgt != pad_index)

        return masks, out_mask


class FNetEncoder(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()

        for k, v in kwargs.items():
            setattr(self, k, v)

        self.src_embed  = Timer(SinusodialPositionalEmbedding(self.srcV, self.model_dim, self.maxI, self.dropout, self.pad_index))
        
        self.layers = nn.ModuleList([FNetEncoderLayer(**kwargs) for n in range(self.encL)])

        self.lnorm = Timer(LayerNormalization(self.model_dim))

    def __call__(self, src, src_mask=None, tgt_mask=None):

        B = src.shape[0]
        J = src.shape[1]

        J_act = (src_mask == False).to(torch.float).sum(-1).squeeze().view(B)

        W_encoder_dft = FNet.calculate_dft_matrix(B, J, J_act)

        h = self.src_embed(src)

        for layer in self.layers:

            h = layer(h, W_encoder_dft, src_mask=src_mask)

        h = self.lnorm(h)

        return h


class FNetDecoder(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()

        for k, v in kwargs.items():
            setattr(self, k, v)

        self.tgt_embed = Timer(SinusodialPositionalEmbedding(self.tgtV, self.model_dim, self.maxI, self.dropout, self.pad_index))

        self.layers = nn.ModuleList([FNetDecoderLayer(**kwargs) for n in range(self.decL)])

        self.lnorm              = Timer(LayerNormalization(self.model_dim))
        self.output_projection  = Timer(nn.Linear(self.model_dim, self.tgtV))
        self.log_softmax        = Timer(nn.LogSoftmax(dim=-1))

    def __call__(self, tgt, h, i=None, src_mask=None, tgt_mask=None):

        B = tgt.shape[0]
        I = tgt.shape[1]

        J_act = (src_mask == False).to(torch.float).sum(-1).squeeze().view(B)
        I_act = (J_act * self.sentence_length_factor).to(torch.int)

        if Globals.is_training() or not self.stepwise:
            W_decoder_dft = FNet.calculate_dft_matrix(B, I, I_act)
        else:
            W_decoder_dft = FNet.calculate_dft_matrix(B, torch.clamp(torch.max(I_act), min=i), I_act)

        s = self.tgt_embed(tgt, J=i)

        for layer in self.layers:
            s = layer(s, h, W_decoder_dft, src_mask=src_mask, tgt_mask=tgt_mask)

        s = self.lnorm(s)
        s = self.output_projection(s)
        s = self.log_softmax(s)

        return s


class FNetEncoderLayer(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()

        for k, v in kwargs.items():
            setattr(self, k, v)
        
        self.lnorm1     = Timer(LayerNormalization(self.model_dim))
        if self.fnet_enc_self_att:
            self.att    = FNetAttentionLayer(self.model_dim, self.maxI, self.dropout, False, self.just_like_fnet)
        else:
            self.att    = MultiHeadAttention(self.nHeads, self.model_dim, self.dropout)
        self.dropout    = Timer(nn.Dropout(self.dropout))

        self.lnorm2     = Timer(LayerNormalization(self.model_dim))
        self.ff1        = Timer(nn.Linear(self.model_dim, self.ff_dim))
        self.relu       = Timer(nn.ReLU())
        self.ff2        = Timer(nn.Linear(self.ff_dim, self.model_dim))

    def __call__(self, x, W_encoder_dft, src_mask=None):

        J = x.shape[1]
        
        r = x
        x = self.lnorm1(x)
        if self.fnet_enc_self_att:
            x, _ = self.att(x, W_encoder_dft, J, m=src_mask)
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


class FNetDecoderLayer(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()

        for k, v in kwargs.items():
            setattr(self, k, v)

        self.lnorm1         = Timer(LayerNormalization(self.model_dim))
        if self.fnet_dec_self_att:
            self.self_att   = FNetAttentionLayer(self.model_dim, self.maxI, self.dropout, self.stepwise, self.just_like_fnet)
        else:
            self.self_att   = MultiHeadAttention(self.nHeads, self.model_dim, self.dropout)
        self.self_att_state = DynamicState(time_dim=1, stepwise=self.stepwise)

        self.lnorm2         = Timer(LayerNormalization(self.model_dim))
        self.cross_att  = MultiHeadAttention(self.nHeads, self.model_dim, self.dropout)

        self.lnorm3     = Timer(LayerNormalization(self.model_dim))
        self.ff1        = Timer(nn.Linear(self.model_dim, self.ff_dim))
        self.relu       = Timer(nn.ReLU())
        self.ff2        = Timer(nn.Linear(self.ff_dim, self.model_dim))
        self.dropout    = Timer(nn.Dropout(self.dropout))

    def __call__(self, s, h, W_decoder_dft, src_mask=None, tgt_mask=None):
        
        r = s
        s = self.lnorm1(s)
        s_full = self.self_att_state.full(s)
        if self.fnet_dec_self_att:
            s, _ = self.self_att(s_full, W_decoder_dft, s_full.shape[1], m=tgt_mask)
        else:
            s, _ = self.self_att(s, s_full, s_full, m=tgt_mask)
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


class FNetAttentionLayer(nn.Module):

    def __init__(self, D, maxI, dropout, stepwise, just_like_fnet):
        super().__init__()

        self.stepwise = stepwise
        self.just_like_fnet = just_like_fnet
        
        self.W_v = nn.Linear(D, D)

        if not self.just_like_fnet:
            self.W_o = nn.Linear(D, D)

        self.dropout = nn.Dropout(dropout)
        self.transpose = Timer(Transpose())

    def __call__(self, v, W_dft, L_out, m=None):
        
        L_in = v.shape[1]

        v = self.W_v(v)

        if Globals.is_training() or not self.stepwise:
            a = W_dft[:,:L_out,:]
        else:
            a = W_dft[:,L_out-1,:L_in]
            a = a.unsqueeze(1)

        if m is not None:
            m = m.squeeze(1)
            a = a.masked_fill(m, 0.)

        v = torch.complex(v, torch.zeros_like(v))

        o = torch.matmul(a, v)

        if self.just_like_fnet:
            o = torch.fft.fft(o, dim=-1)
            o = torch.real(o)
            o = self.dropout(o)

        else:
            o = torch.real(o)
            o = self.dropout(o)
            o = self.W_o(o)

        return o, a