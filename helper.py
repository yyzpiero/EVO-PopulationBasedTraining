import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions.categorical import Categorical
import functools
from typing import Any, Dict

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

def matrix_norm(v, axis=1):
    
    if np.all(v==0):
        return v
    norm = np.divide(v , np.tile(np.sum(v, axis), (v.shape[axis], axis)).transpose())
    
    return norm

class FastGLU(nn.Module):
    def __init__(self, in_size):
        super().__init__()
        self.in_size = in_size
        self.linear = layer_init(nn.Linear(in_size, in_size * 2))
    
    def forward(self, x):
        x = self.linear(x)
        out = x[:, self.in_size:] * x[:, :self.in_size].sigmoid()
        return out


class Transformer(nn.Module):
    ''' AlphaStar transformer composed with only three encoder layers '''
    # refactored by reference to https://github.com/metataro/sc2_imitation_learning

    # default parameter from AlphaStar
    def __init__(
            self, d_model=256, d_inner=1024,
            n_layers=3, n_head=2, d_k=128, d_v=128, dropout=0.1):

        super().__init__()

        self.encoder = Encoder(
            d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            dropout=dropout)

        # for p in self.parameters():
        #     if p.dim() > 1:
        #         nn.init.xavier_uniform_(p)

    def forward(self, x, mask=None):
        enc_output, *_ = self.encoder(x, mask=mask)

        return enc_output


class Encoder(nn.Module):
    ''' A alphastar encoder model with self attention mechanism. '''

    # default parameter from AlphaStar
    def __init__(
            self, n_layers=3, n_head=2, d_k=128, d_v=128,
            d_model=256, d_inner=1024, dropout=0.1):

        super().__init__()

        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])

        # note "unbiased=False" will affect the results
        # layer_norm is b = (a - torch.mean(a))/(torch.var(a, unbiased=False)**0.5) * 1.0 + 0.0

    def forward(self, x, mask=None):
        # -- Forward

        enc_output = x
        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output, mask=mask)

        del enc_slf_attn

        return enc_output,


class EncoderLayer(nn.Module):
    '''     
    '''

    # default parameter from AlphaStar
    def __init__(self, d_model=256, d_inner=1024, n_head=2, d_k=128, d_v=128, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)

        self.ln1 = nn.LayerNorm(d_model, eps=1e-6)
        self.ln2 = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, x, mask=None):
        att_out, enc_slf_attn = self.slf_attn(x, x, x, mask=mask)

        att_out = self.drop1(att_out)
        out_1 = self.ln1(x + att_out)

        ffn_out = self.pos_ffn(out_1)

        ffn_out = self.drop2(ffn_out)
        out = self.ln2(out_1 + ffn_out)

        del att_out, out_1, ffn_out

        return out, enc_slf_attn

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1, bias_value=-1e9):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.biasval = bias_value

    def forward(self, q, k, v, mask=None):

        # q: (b, n, lq, dk)
        # k: (b, n, lk, dk)
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        # atten: (b, n, lq, lk),
        if mask is not None:
            attn = attn.masked_fill(mask == 0, self.biasval)
            del mask

        attn = self.dropout(F.softmax(attn, dim=-1))

        # v: (b, n, lv, dv)
        # r: (b, n, lq, dv)
        r = torch.matmul(attn, v)

        return r, attn


class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        # pre-attention projection
        self.w_qs = layer_init(nn.Linear(d_model, n_head * d_k, bias=True))
        self.w_ks = layer_init(nn.Linear(d_model, n_head * d_k, bias=True))
        self.w_vs = layer_init(nn.Linear(d_model, n_head * d_v, bias=True))

        # after-attention projection
        self.fc = layer_init(nn.Linear(n_head * d_v, d_model, bias=True))

        # attention
        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

    def forward(self, q, k, v, mask=None):
        # q: (b, lq, dm)
        # k: (b, lk, dm)
        # v: (b, lv, dm)

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        size_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        # pass through the pre-attention projection
        # separate different heads

        # after that q: (b, lq, n, dk)
        q = self.w_qs(q).view(size_b, len_q, n_head, d_k)

        k = self.w_ks(k).view(size_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(size_b, len_v, n_head, d_v)

        # transpose for attention dot product: (b, n, lq, dk)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.

        q, attn = self.attention(q, k, v, mask=mask)

        # q: (b, n, lq, dk), k: (b, n, lk, dk), atten = q \matmul k^t = (b, n, lq, lk),
        # v: (b, n, lv, dv), assert lk = lv
        # atten \matmul v = (b, n, lq, dv)

        # transpose to move the head dimension back: (b, lq, n, dv)
        # combine the last two dimensions to concatenate all the heads together: (b, lq, (n*dv))
        q = q.transpose(1, 2).contiguous().view(size_b, len_q, -1)

        # q: (b, lq, (n*dv)) \matmul ((n*dv), dm) = (b, lq, dm)
        # note, q has the same shape as when it enter in
        q = self.fc(q)

        del mask, k, v, 

        return q, attn


class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = layer_init(nn.Linear(d_in, d_hid))  # position-wise
        self.w_2 = layer_init(nn.Linear(d_hid, d_in))  # position-wise

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)

        return x

class CategoricalMasked(Categorical):
    def __init__(self, probs=None, logits=None, validate_args=None, masks=[], device = "cpu"):
        self.masks = masks
        self.device = device
        if len(self.masks) == 0:
            super(CategoricalMasked, self).__init__(probs, logits, validate_args)
        else:
            self.masks = masks.type(torch.BoolTensor).to(self.device)
            logits = torch.where(self.masks, logits, torch.tensor(-1e17).to(self.device))
            super(CategoricalMasked, self).__init__(probs, logits, validate_args)

    def entropy(self):
        if len(self.masks) == 0:
            return super(CategoricalMasked, self).entropy()
        p_log_p = self.logits * self.probs
        p_log_p = torch.where(self.masks, p_log_p, torch.tensor(0.0).to(self.device))
        return -p_log_p.sum(-1)



def recursive_getattr(obj: Any, attr: str, *args) -> Any:
    """
    Recursive version of getattr
    taken from https://stackoverflow.com/questions/31174295
    Ex:
    > MyObject.sub_object = SubObject(name='test')
    > recursive_getattr(MyObject, 'sub_object.name')  # return test
    :param obj:
    :param attr: Attribute to retrieve
    :return: The attribute
    """

    def _getattr(obj: Any, attr: str) -> Any:
        return getattr(obj, attr, *args)

    return functools.reduce(_getattr, [obj] + attr.split("."))