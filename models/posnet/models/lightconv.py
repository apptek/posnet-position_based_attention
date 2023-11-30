
import torch
import torch.nn as nn
from numpy import dtype

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

@register_model("LightConv")
class LightConv(MTModel):

    def __init__(self, **kwargs):
        super().__init__()

        for k, v in kwargs.items():
            setattr(self, k, v)

        self.encoder = LightConvEncoder(**kwargs)
        self.decoder = LightConvDecoder(**kwargs)

        if self.tiew:
            self.decoder.tgt_embed.object_to_time.word_embed.weight = self.encoder.src_embed.object_to_time.word_embed.weight
            self.decoder.output_projection.object_to_time.weight = self.encoder.src_embed.object_to_time.word_embed.weight

    @staticmethod
    def create_model_from_config(config, vocab_src, vocab_tgt):
        
        model =  LightConv(
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
            use_sinusodial_pos_embed = config['use_sinusodial_pos_embed', True],
            use_pos_embed = config['use_pos_embed', True],
            K = config['K'],
            K_per_layer = config['K_per_layer', False],
            Ks = config['Ks', []],
            gating_v = config['gating_v'],
            gating_g = config['gating_g'],
            normalize_v = config['normalize_v'],
            normalize_g = config['normalize_g'],
            global_context = config['global_context', False]
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

    def get_matrices(self, src, tgt, src_mask=None, tgt_mask=None):
        
        h, enc_matrices = self.encoder.get_matrices(src, src_mask=src_mask, tgt_mask=tgt_mask)

        s, dec_matrices = self.decoder.get_matrices(tgt, h, src_mask=src_mask, tgt_mask=tgt_mask)

        enc_matrices.update(dec_matrices)

        return s, enc_matrices

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


class LightConvEncoder(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()

        for k, v in kwargs.items():
            setattr(self, k, v)

        if self.use_sinusodial_pos_embed:
            self.src_embed = Timer(SinusodialPositionalEmbedding(self.srcV, self.model_dim, self.maxI, self.dropout, self.pad_index))
        else:
            self.src_embed = Timer(PositionalEmbedding(self.srcV, self.model_dim, self.maxI, self.dropout, self.pad_index))
        
        if self.K_per_layer:

            import copy
            args_per_layer = []
            for l in range(self.encL):
                args = copy.deepcopy(kwargs)
                args['K'] = kwargs['Ks'][l]
                args_per_layer.append(args)

            self.layers = nn.ModuleList([LightConvEncoderLayer(**args_per_layer[l]) for l in range(self.encL)])
        else:
            self.layers = nn.ModuleList([LightConvEncoderLayer(**kwargs) for l in range(self.encL)])

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


class LightConvDecoder(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()

        for k, v in kwargs.items():
            setattr(self, k, v)

        if self.use_sinusodial_pos_embed:
            self.tgt_embed = Timer(SinusodialPositionalEmbedding(self.tgtV, self.model_dim, self.maxI, self.dropout, self.pad_index))
        else:
            self.tgt_embed = Timer(PositionalEmbedding(self.tgtV, self.model_dim, self.maxI, self.dropout, self.pad_index))

        if self.K_per_layer:
            import copy
            args_per_layer = []
            for l in range(self.decL):
                args = copy.deepcopy(kwargs)
                args['K'] = kwargs['Ks'][l]
                args_per_layer.append(args)

            self.layers = nn.ModuleList([LightConvDecoderLayer(**args_per_layer[l]) for l in range(self.decL)])
        else:
            self.layers = nn.ModuleList([LightConvDecoderLayer(**kwargs) for l in range(self.decL)])

        self.lnorm              = Timer(LayerNormalization(self.model_dim))
        self.output_projection  = Timer(nn.Linear(self.model_dim, self.tgtV))
        self.log_softmax        = Timer(nn.LogSoftmax(dim=-1))

    def __call__(self, tgt, h, i=None, src_mask=None, tgt_mask=None):

        s = self.tgt_embed(tgt, J=i)

        for layer in self.layers:

            s, _, _ = layer(s, h, i_scalar=i, src_mask=src_mask, tgt_mask=tgt_mask)

        s = self.lnorm(s)
        s = self.output_projection(s)
        s = self.log_softmax(s)

        return s

    def get_matrices(self, tgt, h, i=None, src_mask=None, tgt_mask=None):

        s = self.tgt_embed(tgt, J=i)

        matrices = {}

        for l, layer in enumerate(self.layers):

            s, b, c = layer(s, h, i_scalar=i, src_mask=src_mask, tgt_mask=tgt_mask)

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


class LightConvEncoderLayer(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()

        for k, v in kwargs.items():
            setattr(self, k, v)
        
        self.lnorm1     = Timer(LayerNormalization(self.model_dim))
        self.att        = LightConvLayer(
            self.nHeads, self.model_dim, self.K, self.dropout, self.maxI,
            gating_v=self.gating_v,
            gating_g=self.gating_g,
            normalize_v=self.normalize_v,
            normalize_g=self.normalize_g,
            global_context=self.global_context
        )
        self.dropout    = Timer(nn.Dropout(self.dropout))

        self.lnorm2     = Timer(LayerNormalization(self.model_dim))
        self.ff1        = Timer(nn.Linear(self.model_dim, self.ff_dim))
        self.relu       = Timer(nn.ReLU())
        self.ff2        = Timer(nn.Linear(self.ff_dim, self.model_dim))

    def __call__(self, x, src_mask=None):
        
        r = x
        x = self.lnorm1(x)
        x, a = self.att(x, x, m=src_mask)
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


class LightConvDecoderLayer(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()

        for k, v in kwargs.items():
            setattr(self, k, v)

        self.lnorm1         = Timer(LayerNormalization(self.model_dim))
        self.self_att       = LightConvLayer(
            self.nHeads, self.model_dim, self.K, self.dropout, self.maxI, stepwise=self.stepwise,
            gating_v=self.gating_v,
            gating_g=self.gating_g,
            normalize_v=self.normalize_v,
            normalize_g=self.normalize_g,
            global_context=self.global_context
        )
        self.self_att_state = DynamicState(time_dim=1, stepwise=self.stepwise)

        self.lnorm2     = Timer(LayerNormalization(self.model_dim))
        self.cross_att  = MultiHeadAttention(self.nHeads, self.model_dim, self.dropout)

        self.lnorm3     = Timer(LayerNormalization(self.model_dim))
        self.ff1        = Timer(nn.Linear(self.model_dim, self.ff_dim))
        self.relu       = Timer(nn.ReLU())
        self.ff2        = Timer(nn.Linear(self.ff_dim, self.model_dim))
        self.dropout    = Timer(nn.Dropout(self.dropout))

    def __call__(self, s, h, i_scalar=None, src_mask=None, tgt_mask=None):

        r = s
        s = self.lnorm1(s)
        s_full = self.self_att_state.full(s)
        s, b = self.self_att(s_full, s, i_scalar=i_scalar, m=tgt_mask)
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


class LightConvLayer(nn.Module):

    def __init__(self,
        H, D, K, dropout, maxI,
        stepwise=False,
        gating_v='gelu',
        gating_g='gelu',
        normalize_v=True,
        normalize_g=False,
        global_context=False
    ):
        super().__init__()

        self.H = H
        self.D = D
        self.Dh = D // H
        self.K = K

        self.stepwise = stepwise
        self.gating_v = gating_v
        self.gating_g = gating_g
        self.normalize_v = normalize_v
        self.normalize_g = normalize_g
        self.global_context = global_context

        self.__create_gating_activations(gating_v, gating_g)
        self.__create_normalizations(gating_v, normalize_v, gating_g, normalize_g)
        self.__create_learnable_parameters(H, D, K, gating_v, gating_g)

        self.transpose = Timer(Transpose())
        self.softmax = Timer(nn.Softmax(-1))
        self.dropout = Timer(nn.Dropout(dropout))
        self.matmul = Timer(MatMul())

        self.__precalculate_indices_and_mask(K, H, maxI)

    def __create_gating_activations(self, gating_v, gating_g):

        if gating_v == 'glu':
            self.act_v = nn.GLU()
        elif gating_v == 'gelu':
            self.act_v = nn.GELU()
        elif gating_v == 'none':
            self.act_v = nn.Identity()
        else:
            raise ValueError(f'Unrecognized value for gating_v {gating_v}')

        if gating_g == 'glu':
            self.act_g = nn.GLU()
        elif gating_g == 'gelu':
            self.act_g = nn.GELU()
        elif gating_g == 'sigmoid':
            self.act_g = nn.Sigmoid()
        elif gating_g == 'none':
            self.act_g = nn.Identity()
        else:
            raise ValueError(f'Unrecognized value for gating_g {gating_g}')

    def __create_normalizations(self, gating_v, normalize_v, gating_g, normalize_g):

        if gating_v != 'none' and normalize_v:
            self.lnorm_v = Timer(LayerNormalization(self.D))
        else:
            self.lnorm_v = nn.Identity()

        if gating_g != 'none' and normalize_g:
            self.lnorm_g = Timer(LayerNormalization(self.D))
        else:
            self.lnorm_g = nn.Identity()

    def __create_learnable_parameters(self, H, D, K, gating_v, gating_g):

        if gating_v == 'glu':
            self.W_v = Timer(nn.Linear(D, 2*D))
        else:
            self.W_v = Timer(nn.Linear(D, D))

        if gating_g != 'none':
            if gating_g == 'glu':
                self.W_g = Timer(nn.Linear(D, 2*D))
            else:
                self.W_g = Timer(nn.Linear(D, D))
        else:
            self.W_g = nn.Identity()
        
        self.W_o = Timer(nn.Linear(D, D))

        self.W = nn.Parameter(torch.ones(H, (2*K+1)), requires_grad=True)

    def __precalculate_indices_and_mask(self, K, H, maxI):

        arangeI = torch.arange(maxI)
        indices = arangeI.unsqueeze(0)
        indices = indices.repeat(maxI, 1)
        indices = indices - arangeI.unsqueeze(1)
        indices = torch.clamp(indices, max=K, min=-K)
        indices += K
        indices = indices.unsqueeze(0).unsqueeze(0)
        indices = indices.repeat(1, H, 1, 1)

        gather_m = torch.zeros(1,1,maxI,maxI).to(torch.int)
        for i in range(maxI):
            l = max(0, i-K)
            r = min(i+K+1, maxI)
            gather_m[:,:,i,:l] = 1
            gather_m[:,:,i,r:] = 1

        self.register_buffer('indices', indices)
        self.register_buffer('gather_m', gather_m.to(torch.bool))

    def __call__(self, v, g, i_scalar=None, m=None):

        B = v.shape[0]
        I = v.shape[1]
        H = self.H
        K = self.K
        D = self.D
        Dh = self.Dh
        
        v = self.W_v(v)
        g = self.W_g(g)

        v = self.act_v(v)
        g = self.act_g(g)

        v = self.lnorm_v(v)
        g = self.lnorm_g(g)

        v = v.view(B, -1, H, Dh)
        v = self.transpose(v, 1, 2)

        if self.gating_g:
            g = g.view(B, -1, H, Dh)
            g = self.transpose(g, 1, 2)

        if not Globals.is_training() and self.stepwise:
            assert i_scalar is not None
            indices = self.indices[:,:,i_scalar-1,:I].unsqueeze(-2)
            if not self.global_context:
                gather_m = self.gather_m[:,:,i_scalar-1,:I].unsqueeze(-2)
        else:
            indices = self.indices[:,:,:I,:I]
            if not self.global_context:
                gather_m = self.gather_m[:,:,:I,:I]
            
        a = self.W.unsqueeze(1).unsqueeze(0)
        a = a.repeat(1, 1, I, 1)
        a = torch.gather(a, -1, indices)

        if not self.global_context:
            a = a.masked_fill(gather_m, -1e15)

        if m is not None:
            a = a.masked_fill(m, -1e15)

        a = self.softmax(a)
        a = self.dropout(a)

        o = self.matmul(a, v)

        if self.gating_g != 'none':
            o = o * g

        o = self.transpose(o, 1, 2)
        o = o.reshape(B, -1, D)
        o = self.W_o(o)

        return o, a