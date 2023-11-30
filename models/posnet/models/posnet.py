import math

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

@register_model("PosNet")
class PosNet(MTModel):

    def __init__(self, **kwargs):
        super().__init__()

        for k, v in kwargs.items():
            setattr(self, k, v)

        self.encoder = PosNetEncoder(**kwargs)
        self.decoder = PosNetDecoder(**kwargs)

        if self.tiew:
            self.decoder.tgt_embed.word_embed.weight = self.encoder.src_embed.word_embed.weight
            self.decoder.output_projection.weight = self.encoder.src_embed.word_embed.weight

    @staticmethod
    def create_model_from_config(config, vocab_src, vocab_tgt):
        
        model =  PosNet(
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
            posnet_enc_self_att = config['posnet_enc_self_att'],
            posnet_dec_self_att = config['posnet_dec_self_att'],
            posnet_dec_cross_att = config['posnet_dec_cross_att'],
            use_pos_embed = config['use_pos_embed', True],
            posnet_type = config['posnet_type'],
            use_sinusodial_pos_embed = (config['posnet_type'] == 'aPosNet'),
            K_per_layer = config['K_per_layer', False],
            Ks = config['Ks', []],
            gating_v = config['gating_v'],
            gating_g = config['gating_g'],
            gating_o = config['gating_o', 'none'],
            normalize_v = config['normalize_v'],
            normalize_g = config['normalize_g'],
            cross_gating_v = config['cross_gating_v', 'glu'],
            cross_gating_g = config['cross_gating_g', 'none'],
            cross_gating_o = config['cross_gating_o', 'none'],
            cross_normalize_v = config['cross_normalize_v', False],
            cross_normalize_g = config['cross_normalize_g', False],
            pre_calculate_matrices = config['pre_calculate_matrices', True],
            length_ratio = config['length_ratio']
        )

        return model

    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    # Override to pre-calculate matrices
    def load_state_dict(self, *args, **kwargs):

        if not self.pre_calculate_matrices:
            MTModel.load_state_dict(self, *args, **kwargs)
            return

        state_dict = args[0]
        own_state_dict = self.state_dict()

        for name, param in state_dict.items():
            if isinstance(param, torch.nn.Parameter):
                param = param.data
            own_state_dict[name].copy_(param)

        self.eval()

        self.__precalculate_matrices_and_remove_parameters()

    def __precalculate_matrices_and_remove_parameters(self):

        my_print('Pre-calcualting attention matrices!')

        B = 1
        I = self.maxI

        src = torch.arange(B*I).view(B, I)
        tgt = torch.arange(B*I).view(B, I)

        _, j = self.encoder._embed(src)
        _, i = self.decoder._embed(tgt)

        posatt_layers = {}

        for name, module in self.named_modules():
            if isinstance(module, PositionBasedAttention):
                posatt_layers[name] = module
        
        for k, l in posatt_layers.items():

            posnet_type = l.posnet_type

            if posnet_type == 'aPosNet' or posnet_type == 'arPosNet':
                if 'encoder' in k:
                    aa = l._calculate_aposnet_matrix(j, j)
                elif 'decoder' in k and 'self_att' in k:
                    aa = l._calculate_aposnet_matrix(i, i)
                elif 'decoder' in k and 'cross_att' in k:
                    aa = l._calculate_aposnet_matrix(i, j)
                else:
                    raise ValueError(f'Unrecognized layer type {k}')

                l.Wa = nn.Parameter(aa, requires_grad=True)

                delattr(l, 'W_q1')
                delattr(l, 'W_k')

            if posnet_type == 'rPosNet' or posnet_type == 'arPosNet':
                if 'encoder' in k:
                    ar = l._calculate_rposnet_matrix(j)
                elif 'decoder' in k and 'self_att' in k:
                    ar = l._calculate_rposnet_matrix(i)
                elif 'decoder' in k and 'cross_att' in k:
                    ar = l._calculate_rposnet_matrix(i)
                else:
                    raise ValueError(f'Unrecognized layer type {k}')

                l.Wr = nn.Parameter(ar, requires_grad=True)

                delattr(l, 'W_q2')
                delattr(l, 'rel_embed')
                delattr(l, 'rng')

    def __call__(self, src, tgt, src_mask=None, tgt_mask=None):
        
        h, j = self.encoder(src, src_mask=src_mask, tgt_mask=tgt_mask)

        s = self.decoder(tgt, h, j, src_mask=src_mask, tgt_mask=tgt_mask)

        return s, h

    def get_matrices(self, src, tgt, src_mask=None, tgt_mask=None):
        
        h, j, enc_matrices = self.encoder.get_matrices(src, src_mask=src_mask, tgt_mask=tgt_mask)

        s, dec_matrices = self.decoder.get_matrices(tgt, h, j, src_mask=src_mask, tgt_mask=tgt_mask)

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


class PosNetEncoder(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()

        for k, v in kwargs.items():
            setattr(self, k, v)

        if self.use_sinusodial_pos_embed:
            self.src_embed = SinusodialPositionalEmbedding(
                self.srcV, self.model_dim, self.maxI, self.dropout, self.pad_index,
                use_pos_embed=self.use_pos_embed,
                return_pos_embed=True
            )
        else:
            self.src_embed = PositionalEmbedding(
                self.srcV, self.model_dim, self.maxI, self.dropout, self.pad_index,
                use_pos_embed=self.use_pos_embed,
                return_pos_embed=True
            )
        
        if self.K_per_layer:

            import copy
            args_per_layer = []
            for l in range(self.encL):
                args = copy.deepcopy(kwargs)
                args['K'] = kwargs['Ks'][l]
                args_per_layer.append(args)

            self.layers = nn.ModuleList([PosNetEncoderLayer(**args_per_layer[l]) for l in range(self.encL)])
        else:
            self.layers = nn.ModuleList([PosNetEncoderLayer(**kwargs) for l in range(self.encL)])
        
        if self.posnet_enc_self_att or self.posnet_dec_cross_att:
            if not self.use_sinusodial_pos_embed:
                self.lnorm1 = LayerNormalization(self.model_dim)
            self.dropout = nn.Dropout(self.dropout)

        self.lnorm2  = LayerNormalization(self.model_dim)

    def __call__(self, src, src_mask=None, tgt_mask=None):

        h, j = self._embed(src, src_mask=src_mask)

        for layer in self.layers:
            h, _ = layer(h, j, src_mask=src_mask)

        h = self.lnorm2(h)

        return h, j

    def get_matrices(self, src, src_mask=None, tgt_mask=None):

        h, j = self._embed(src, src_mask=src_mask)

        matrices = {}

        for l, layer in enumerate(self.layers):

            h, a = layer(h, j, src_mask=src_mask)

            matrices[f'encoder/layer{l}/self_att'] = {
                'value': a
            }

        h = self.lnorm2(h)

        return h, j, matrices

    def _embed(self, src, src_mask=None):

        h, j = self.src_embed(src)

        if self.posnet_enc_self_att or self.posnet_dec_cross_att:
            if not self.use_sinusodial_pos_embed:
                j = self.lnorm1(j)
            j = self.dropout(j)

        return h, j


class PosNetDecoder(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()

        for k, v in kwargs.items():
            setattr(self, k, v)

        if self.use_sinusodial_pos_embed:
            self.tgt_embed = SinusodialPositionalEmbedding(
                self.tgtV, self.model_dim, self.maxI, self.dropout, self.pad_index,
                use_pos_embed=self.use_pos_embed,
                return_pos_embed=True
            )
        else:
            self.tgt_embed = PositionalEmbedding(
                self.tgtV, self.model_dim, self.maxI, self.dropout, self.pad_index,
                use_pos_embed=self.use_pos_embed,
                return_pos_embed=True
            )

        if self.K_per_layer:
            import copy
            args_per_layer = []
            for l in range(self.decL):
                args = copy.deepcopy(kwargs)
                args['K'] = kwargs['Ks'][l]
                args_per_layer.append(args)

            self.layers = nn.ModuleList([PosNetDecoderLayer(**args_per_layer[l]) for l in range(self.decL)])
        else:
            self.layers = nn.ModuleList([PosNetDecoderLayer(**kwargs) for l in range(self.decL)])

        if self.posnet_dec_self_att or self.posnet_dec_cross_att:
            if not self.use_sinusodial_pos_embed:
                self.lnorm1 = LayerNormalization(self.model_dim)
            self.dropout = nn.Dropout(self.dropout)

        self.lnorm2 = LayerNormalization(self.model_dim)
        self.output_projection = nn.Linear(self.model_dim, self.tgtV)
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def __call__(self, tgt, h, j, i=None, src_mask=None, tgt_mask=None):

        i_scalar = i

        s, i = self._embed(tgt, i=i)

        for layer in self.layers:
            s, _, _ = layer(s, i, h, j, i_scalar=i_scalar, src_mask=src_mask, tgt_mask=tgt_mask)

        s = self.lnorm2(s)
        s = self.output_projection(s)
        s = self.log_softmax(s)

        return s

    def get_matrices(self, tgt, h, j, i=None, src_mask=None, tgt_mask=None):

        i_scalar = i
        matrices = {}

        s, i = self._embed(tgt, J=i)

        for l, layer in enumerate(self.layers):

            s, b, c = layer(s, i, h, j, i_scalar=i_scalar, src_mask=src_mask, tgt_mask=tgt_mask)

            matrices[f'decoder/layer{l}/self_att'] = {
                'value': b
            }

            matrices[f'decoder/layer{l}/cross_att'] = {
                'value': c
            }

        s = self.lnorm2(s)
        s = self.output_projection(s)
        s = self.log_softmax(s)

        return s, matrices

    def _embed(self, tgt, i=None):

        s, i = self.tgt_embed(tgt, J=i)

        if self.posnet_dec_self_att or self.posnet_dec_cross_att:
            if not self.use_sinusodial_pos_embed:
                i = self.lnorm1(i)
            i = self.dropout(i)

        return s, i


class PosNetEncoderLayer(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()

        for k, v in kwargs.items():
            setattr(self, k, v)
        
        self.lnorm1 = LayerNormalization(self.model_dim)
        if self.posnet_enc_self_att:
            self.att = PositionBasedAttention(
                self.nHeads, self.model_dim, self.K, self.maxI, self.dropout, self.posnet_type,
                gating_v=self.gating_v,
                gating_g=self.gating_g,
                gating_o=self.gating_o,
                normalize_v=self.normalize_v,
                normalize_g=self.normalize_g,
                pre_calculate_matrices = self.pre_calculate_matrices
            )
        else:
            self.att = MultiHeadAttention(self.nHeads, self.model_dim, self.dropout)
        self.dropout = nn.Dropout(self.dropout)

        self.lnorm2 = LayerNormalization(self.model_dim)
        self.ff1 = nn.Linear(self.model_dim, self.ff_dim)
        self.relu = nn.ReLU()
        self.ff2 = nn.Linear(self.ff_dim, self.model_dim)

    def __call__(self, x, j, src_mask=None):

        r = x
        x = self.lnorm1(x)
        if self.posnet_enc_self_att:
            x, a = self.att(j, j, x, x, m=src_mask)
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


class PosNetDecoderLayer(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()

        for k, v in kwargs.items():
            setattr(self, k, v)

        self.lnorm1 = LayerNormalization(self.model_dim)
        if self.posnet_dec_self_att:
            self.self_att = PositionBasedAttention(
                self.nHeads, self.model_dim, self.K, self.maxI, self.dropout, self.posnet_type,
                stepwise=self.stepwise,
                gating_v=self.gating_v,
                gating_g=self.gating_g,
                gating_o=self.gating_o,
                normalize_v=self.normalize_v,
                normalize_g=self.normalize_g,
                pre_calculate_matrices = self.pre_calculate_matrices
            )
        else:
            self.self_att = MultiHeadAttention(self.nHeads, self.model_dim, self.dropout)
        self.self_att_state_s = DynamicState(time_dim=1, stepwise=self.stepwise)

        if self.posnet_dec_cross_att:
            if self.cross_gating_g != 'none':
                self.lnorm2 = LayerNormalization(self.model_dim)
            self.cross_att = PositionBasedAttention(
                self.nHeads, self.model_dim, self.K, self.maxI, self.dropout, self.posnet_type,
                stepwise=self.stepwise,
                gating_v=self.cross_gating_v,
                gating_g=self.cross_gating_g,
                gating_o=self.cross_gating_o,
                normalize_v=self.cross_normalize_v,
                normalize_g=self.cross_normalize_g,
                length_ratio=self.length_ratio
            )
        else:
            self.lnorm2 = LayerNormalization(self.model_dim)
            self.cross_att = MultiHeadAttention(self.nHeads, self.model_dim, self.dropout)

        self.lnorm3 = LayerNormalization(self.model_dim)
        self.ff1 = nn.Linear(self.model_dim, self.ff_dim)
        self.relu = nn.ReLU()
        self.ff2 = nn.Linear(self.ff_dim, self.model_dim)
        self.dropout = nn.Dropout(self.dropout)

    def __call__(self, s, i, h, j, i_scalar=None, src_mask=None, tgt_mask=None):

        r = s
        s = self.lnorm1(s)
        s_full = self.self_att_state_s.full(s)
        if self.posnet_dec_self_att:
            s, b = self.self_att(i, i, s_full, s, i_scalar=i_scalar, m=tgt_mask)
        else:
            s, b = self.self_att(s, s_full, s_full, m=tgt_mask)
        s = self.dropout(s)
        s = s + r

        r = s
        if self.posnet_dec_cross_att:
            if self.cross_gating_g != 'none':
                s = self.lnorm2(s)
            s, c = self.cross_att(i, j, h, s, i_scalar=i_scalar, m=src_mask)
        else:
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


class PositionBasedAttention(nn.Module):

    def __init__(self, H, D, K, maxI, dropout, posnet_type,
        stepwise=False,
        gating_v='gelu',
        gating_g='gelu',
        gating_o='none',
        normalize_v=True,
        normalize_g=False,
        pre_calculate_matrices=True,
        length_ratio=1.0
    ):
        super().__init__()

        self.H = H
        self.D = D
        self.Dh = D // H
        self.K = K

        self.stepwise = stepwise
        self.posnet_type = posnet_type
        self.gating_v = gating_v
        self.gating_g = gating_g
        self.gating_o = gating_o
        self.normalize_v = normalize_v
        self.normalize_g = normalize_g
        self.pre_calculate_matrices = pre_calculate_matrices

        self.__create_activations(gating_v, gating_g, gating_o)
        self.__create_normalizations(gating_v, normalize_v, gating_g, normalize_g)
        self.__create_learnable_parameters(H, D, K, maxI, posnet_type, gating_v, gating_g)

        self.transpose = Transpose()
        self.softmax = nn.Softmax(-1)
        self.dropout = nn.Dropout(dropout)
        self.matmul = MatMul()

        self.__precalculate_indices(K, H, maxI, length_ratio)

    def __create_activations(self, gating_v, gating_g, gating_o):

        if gating_v == 'glu':
            self.act_v = nn.GLU()
        elif gating_v == 'gelu':
            self.act_v = nn.GELU()
        elif gating_v == 'sigmoid':
            self.act_v = nn.Sigmoid()
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

        if gating_o == 'sigmoid':
            self.act_o = nn.Sigmoid()
        elif gating_o == 'none':
            self.act_o = nn.Identity()
        else:
            raise ValueError(f'Unrecognized value for gating_o {gating_o}')

    def __create_normalizations(self, gating_v, normalize_v, gating_g, normalize_g):

        if gating_v != 'none' and normalize_v:
            self.lnorm_v = LayerNormalization(self.D)
        else:
            self.lnorm_v = nn.Identity()

        if gating_g != 'none' and normalize_g:
            self.lnorm_g = LayerNormalization(self.D)
        else:
            self.lnorm_g = nn.Identity()

    def __create_learnable_parameters(self, H, D, K, maxI, posnet_type, gating_v, gating_g):

        if posnet_type == 'aPosNet' or posnet_type == 'arPosNet':
            self.W_q1 = nn.Linear(D, D)
            self.W_k  = nn.Linear(D, D)
            if not Globals.is_training() and self.pre_calculate_matrices:
                self.Wa = nn.Parameter(torch.ones(1, H, maxI, maxI), requires_grad=True)

        if posnet_type == 'rPosNet' or posnet_type == 'arPosNet':
            self.W_q2 = nn.Linear(D, D)
            self.rel_embed = nn.Embedding(2*K+1, D)
            self.register_buffer('rng', torch.arange(2*K+1))
            if not Globals.is_training() and self.pre_calculate_matrices:
                self.Wr = nn.Parameter(torch.ones(1, H, maxI, 2*K+1), requires_grad=True)

        if gating_v == 'glu':
            self.W_v = nn.Linear(D, 2*D)
        else:
            self.W_v = nn.Linear(D, D)

        self.W_o = nn.Linear(D, D)
        
        if gating_g != 'none':
            if gating_g == 'glu':
                self.W_g = nn.Linear(D, 2*D)
            else:
                self.W_g = nn.Linear(D, D)
        else:
            self.W_g = nn.Identity()

    def __precalculate_indices(self, K, H, maxI, length_ratio):

        arangeI = torch.arange(maxI).to(Globals.get_device())
        indices = arangeI.unsqueeze(0)
        indices = indices.repeat(maxI, 1)
        indices = indices - torch.floor(length_ratio * arangeI.to(torch.float).unsqueeze(1))
        indices = torch.clamp(indices, max=K, min=-K)
        indices += K
        indices = indices.unsqueeze(0).unsqueeze(0).to(torch.int64)

        self.register_buffer('indices', indices)

    def __call__(self, q, k, v, g, i_scalar=None, m=None):
        
        B = v.shape[0]
        J = v.shape[1]
        I = q.shape[1]
        H = self.H
        K = self.K
        D = self.D
        Dh = self.Dh
        posnet_type = self.posnet_type

        q_in = q

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

        a = self.__get_matrix(q_in, k, i_scalar, B, H, I, J)

        a = a / math.sqrt(Dh)

        if m is not None:
            a = a.masked_fill(m, -float('inf'))

        a = self.softmax(a)
        a = self.dropout(a)

        o = self.matmul(a, v)

        o = self.act_o(o)

        if self.gating_g != 'none':
            o = o * g

        o = self.transpose(o, 1, 2)
        o = o.reshape(B, -1, D)
        o = self.W_o(o)

        return o, a

    def __get_matrix(self, q_in, k, i_scalar, B, H, I, J):

        posnet_type = self.posnet_type

        if posnet_type == 'aPosNet' or posnet_type == 'arPosNet':
            
            aa = self.__get_aPosNet_matrix(q_in, k, i_scalar, I, J)
            self.__check_matrix_shape(aa, B, I, J)

        if posnet_type == 'rPosNet' or posnet_type == 'arPosNet':
            
            ar = self.__get_rPosNet_matrix(q_in, i_scalar, B, I, J)
            self.__check_matrix_shape(ar, B, I, J)

        if posnet_type == 'aPosNet':
            a = aa
        elif posnet_type == 'rPosNet':
            a = ar
        elif posnet_type == 'arPosNet':
            a = aa + ar

        return a

    def __get_aPosNet_matrix(self, q_in, k, i_scalar, I, J):

        if not Globals.is_training() and self.pre_calculate_matrices:
            if self.stepwise:
                aa = self.Wa[:,:,i_scalar-1,:J].unsqueeze(-2)
            else:
                aa = self.Wa[:,:,:I,:J]
        else:
            aa = self._calculate_aposnet_matrix(q_in, k)
        
        return aa

    def __get_rPosNet_matrix(self, q_in, i_scalar, B, I, J):

        if not Globals.is_training() and self.pre_calculate_matrices:
            if self.stepwise:
                ar = self.Wr[:,:,i_scalar-1,:].unsqueeze(-2)
            else:
                ar = self.Wr[:,:,:I,:]
        else:
            ar = self._calculate_rposnet_matrix(q_in)

        if not Globals.is_training() and self.stepwise:
            indices = self.indices[:,:,i_scalar-1,:J].unsqueeze(-2)
        else:
            indices = self.indices[:,:,:I,:J]

        if Globals.is_training():
            indices = indices.repeat(B, self.H, 1, 1)
        else:
            indices = indices.repeat(1, self.H, 1, 1)

        ar = torch.gather(ar, -1, indices) # [B, H, I, J]

        return ar

    def _calculate_aposnet_matrix(self, q_in, k):

        B = k.shape[0]
        
        k  = self.W_k(k)
        q = self.W_q1(q_in)

        k = k.view(B, -1, self.H, self.Dh)
        q = q.view(B, -1, self.H, self.Dh)

        k = self.transpose(k, 1, 2)
        q = self.transpose(q, 1, 2)

        return self.matmul(q, self.transpose(k, -2, -1))

    def _calculate_rposnet_matrix(self, q_in):

        B = q_in.shape[0]
        H = self.H
        K = self.K
        Dh = self.Dh

        q = self.W_q2(q_in)
        r = self.rel_embed(self.rng) # [2K+1, D]

        q = q.view(B, -1, H, Dh)
        q = self.transpose(q, 1, 2)

        r = r.view(1,  2*K+1, H, Dh)
        r = self.transpose(r, 1, 2)

        r = self.matmul(q, self.transpose(r, -2, -1)) # [B, H, I, 2K+1]

        return r

    def __check_matrix_shape(self, a, B, I, J):
        if Globals.is_training():
            dim0 = B
            dim2 = I
        else:
            dim0 = 1
            if self.stepwise:
                dim2 = 1
            else:
                dim2 = I

        assert list(a.shape) == [dim0, self.H, dim2, J]