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
    PositionalEmbedding,
    LayerNormalization,
    MultiHeadAttention,
    Transpose,
)

@register_model("GaussianAttn")
class GaussianAttn(MTModel):

    def __init__(self, **kwargs):
        super().__init__()

        for k, v in kwargs.items():
            setattr(self, k, v)

        self.encoder = GaussianAttnEncoder(**kwargs)
        self.decoder = GaussianAttnDecoder(**kwargs)

        if self.tiew:
            self.decoder.tgt_embed.object_to_time.word_embed.weight = self.encoder.src_embed.object_to_time.word_embed.weight
            self.decoder.output_projection.object_to_time.weight = self.encoder.src_embed.object_to_time.word_embed.weight

    @staticmethod
    def create_model_from_config(config, vocab_src, vocab_tgt):
        
        model =  GaussianAttn(
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
            gaussianattn_enc_self_att = config['gaussianattn_enc_self_att'],
            gaussianattn_dec_self_att = config['gaussianattn_dec_self_att'],
            use_multi_head_gaussians = config['use_multi_head_gaussians'],
            use_sinusodial_pos_embed = config['use_sinusodial_pos_embed', False],
            std = config['std', 1.],
            use_glu = config['use_glu', False]
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

    def get_matrices(self, src, tgt, src_mask=None, tgt_mask=None):
        
        h, enc_matrices = self.encoder.get_matrices(src, src_mask=src_mask, tgt_mask=tgt_mask)

        s, dec_matrices = self.decoder.get_matrices(tgt, h, src_mask=src_mask, tgt_mask=tgt_mask)

        enc_matrices.update(dec_matrices)

        return s, enc_matrices


class GaussianAttnEncoder(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()

        for k, v in kwargs.items():
            setattr(self, k, v)

        if self.use_sinusodial_pos_embed:
            self.src_embed = Timer(SinusodialPositionalEmbedding(self.srcV, self.model_dim, self.maxI, self.dropout, self.pad_index))
        else:
            self.src_embed = Timer(PositionalEmbedding(self.srcV, self.model_dim, self.maxI, self.dropout, self.pad_index))
        
        self.layers = nn.ModuleList([GaussianAttnEncoderLayer(**kwargs) for n in range(self.encL)])

        self.lnorm = Timer(LayerNormalization(self.model_dim))

    def __call__(self, src, src_mask=None, tgt_mask=None):

        h = self.src_embed(src)

        for layer in self.layers:

            h, _ = layer(h, src_mask=src_mask)

        h = self.lnorm(h)

        return h

    def get_matrices(self, src, src_mask=None, tgt_mask=None):

        h = self.src_embed(src)

        matrices = {}

        for l, layer in enumerate(self.layers):

            h, a = layer(h, src_mask=src_mask)

            matrices[f'encoder/layer{l}/self_att'] = {
                'value': a
            }

        h = self.lnorm(h)

        return h, matrices


class GaussianAttnDecoder(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()

        for k, v in kwargs.items():
            setattr(self, k, v)

        if self.use_sinusodial_pos_embed:
            self.tgt_embed = Timer(SinusodialPositionalEmbedding(self.tgtV, self.model_dim, self.maxI, self.dropout, self.pad_index))
        else:
            self.tgt_embed = Timer(PositionalEmbedding(self.tgtV, self.model_dim, self.maxI, self.dropout, self.pad_index))

        self.layers = nn.ModuleList([GaussianAttnDecoderLayer(**kwargs) for n in range(self.decL)])

        self.lnorm              = Timer(LayerNormalization(self.model_dim))
        self.output_projection  = Timer(nn.Linear(self.model_dim, self.tgtV))
        self.log_softmax        = Timer(nn.LogSoftmax(dim=-1))

    def __call__(self, tgt, h, i=None, src_mask=None, tgt_mask=None):

        s = self.tgt_embed(tgt, J=i)

        for layer in self.layers:

            s, _, _ = layer(s, h, src_mask=src_mask, tgt_mask=tgt_mask)

        s = self.lnorm(s)
        s = self.output_projection(s)
        s = self.log_softmax(s)

        return s

    def get_matrices(self, tgt, h, i=None, src_mask=None, tgt_mask=None):

        s = self.tgt_embed(tgt, J=i)

        matrices = {}

        for l, layer in enumerate(self.layers):

            s, b, c = layer(s, h, src_mask=src_mask, tgt_mask=tgt_mask)

            matrices[f'decoder/layer{l}/self_att'] = {
                'value': b
            }

            matrices[f'decoder/layer{l}/cross_att'] = {
                'value': c
            }

        s = self.lnorm(s)
        s = self.output_projection(s)
        s = self.log_softmax(s)

        return s, matrices


class GaussianAttnEncoderLayer(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()

        for k, v in kwargs.items():
            setattr(self, k, v)
        
        self.lnorm1     = Timer(LayerNormalization(self.model_dim))
        if self.gaussianattn_enc_self_att:
            self.att    = GaussianMultiHeadAttentionLayer(
                self.nHeads, self.model_dim, self.maxI, self.dropout, False,
                has_multi_head = self.use_multi_head_gaussians,
                offsets = [-1, 0, 1],
                std = self.std,
                use_glu = self.use_glu
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
        if self.gaussianattn_enc_self_att:
            x, a = self.att(x, J, m=src_mask)
        else:
            x, a = self.att(x, x, x, m=src_mask)
        x = self.dropout(x)
        x = x + r

        r = x
        x = self.lnorm2(x)
        x = self.ff1(x)
        x = self.relu(x)
        x = self.ff2(x)
        x = self.dropout(x)
        x = x + r

        return x, a


class GaussianAttnDecoderLayer(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()

        for k, v in kwargs.items():
            setattr(self, k, v)

        self.lnorm1         = Timer(LayerNormalization(self.model_dim))
        if self.gaussianattn_dec_self_att:
            self.self_att   = GaussianMultiHeadAttentionLayer(
                self.nHeads, self.model_dim, self.maxI, self.dropout, self.stepwise,
                has_multi_head = self.use_multi_head_gaussians,
                offsets = [-1, 0],
                std = self.std,
                use_glu = self.use_glu
            )
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

    def __call__(self, s, h, src_mask=None, tgt_mask=None):
        
        r = s
        s = self.lnorm1(s)
        s_full = self.self_att_state.full(s)
        if self.gaussianattn_dec_self_att:
            s, b = self.self_att(s_full, s_full.shape[1], m=tgt_mask)
        else:
            s, b = self.self_att(s, s_full, s_full, m=tgt_mask)
        s = self.dropout(s)
        s = s + r

        r = s
        s = self.lnorm2(s)
        s, c = self.cross_att(s, h, h, m=src_mask)
        s = self.dropout(s)
        s = s + r

        r = s
        s = self.lnorm3(s)
        s = self.ff1(s)
        s = self.relu(s)
        s = self.ff2(s)
        s = self.dropout(s)
        s = s + r

        return s, b, c


class GaussianMultiHeadAttentionLayer(nn.Module):

    def __init__(self,
        H, D, maxI, dropout, stepwise,
        has_multi_head = True,
        offsets = [-1, 0],
        std = 1.,
        use_glu = False
    ):
        super().__init__()

        self.stepwise = stepwise
        self.use_glu = use_glu

        self.H = H
        self.D = D
        self.Dh = D // H
        
        if self.use_glu:
            self.glu = nn.GLU()
            self.W_v = nn.Linear(D, 2*D)
        else:
            self.W_v = nn.Linear(D, D)
        
        self.W_o = nn.Linear(D, D)

        self.dropout = nn.Dropout(dropout)
        self.transpose = Timer(Transpose())

        self.__precalculate_weights(maxI, H, offsets, has_multi_head, std)

    def __precalculate_weights(self, maxI, H, offsets, has_multi_head, std):

        if has_multi_head:
            a = self.__precalculate_multi_head_weights(maxI, offsets, std, H)
        else:
            a = self.__precalculate_single_head_weights(maxI, offsets, std)

        self.register_buffer('W', a)

    def __precalculate_single_head_weights(self, maxI, offsets, std):

        import math

        a = torch.zeros((maxI, maxI)).to(torch.float)

        for offset in offsets:

            diff = (torch.arange(maxI).to(torch.float) - float(offset)) - torch.arange(maxI).to(torch.float).unsqueeze(-1).repeat(1,maxI)

            a += 1/(std * math.sqrt(2*math.pi)) * torch.exp(-0.5 * (diff / std) ** 2)

        a = a.unsqueeze(0).unsqueeze(0)

        return a

    def __precalculate_multi_head_weights(self, maxI, offsets, std, H):

        import math

        matrices = []

        for h in range(H):

            offset = offsets[h % len(offsets)]

            diff = (torch.arange(maxI).to(torch.float) - float(offset)) - torch.arange(maxI).to(torch.float).unsqueeze(-1).repeat(1,maxI)

            a = 1/(std * math.sqrt(2*math.pi)) * torch.exp(-0.5 * (diff / std) ** 2)

            matrices.append(a)

        a = torch.stack(matrices, dim=0)

        a = a.unsqueeze(0)

        return a

    def __call__(self, v, L_out, m=None):
        
        L_in = v.shape[1]
        B = v.shape[0]
        Dh = self.Dh
        D = self.D
        H = self.H

        v = self.W_v(v)
        if self.use_glu:
            v = self.glu(v)
        v = v.view(B, -1, H, Dh)
        v = self.transpose(v, 1, 2)

        if Globals.is_training() or not self.stepwise:
            a = self.W[:,:,:L_out,:L_in]
        else:
            a = self.W[:,:,L_out-1,:L_in]
            a = a.view(1, H, 1, L_in)

        if m is not None:
            a = a.masked_fill(m, 0.)

        a = self.dropout(a)

        o = torch.matmul(a, v)

        o = self.transpose(o, 1, 2)
        o = o.reshape(B, -1, D)
        o = self.W_o(o)

        return o, a