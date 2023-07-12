import functools
import warnings
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor

from multihead_attention import MultiheadAttention


class AMLPCov(MultiheadAttention):

    def __init__(
            self,
            ffn_dimension=16,
            conv_kernel_size=None,
            activation_fn=None,
            add_norm=False,
            *args,
            **kwargs):
        super(AMLPCov, self).__init__(*args, **kwargs)
        self.ffn_dimension = ffn_dimension
        self.add_norm = add_norm
        self.q_landmarks = self._create_landmark()
        self.k_landmarks = self._create_landmark()

        # TODO: remove these temperatures factor and merge them to landmarks
        self.kv_temperature = self._create_temperature()
        self.qq_temperature = self._create_temperature()
        self.kk_temperature = self._create_temperature()
        # self.w_norm = nn.LayerNorm(self.head_dim)
        if self.add_norm:
            
            self.q_norm, self.w_norm = nn.LayerNorm(self.head_dim), nn.LayerNorm(self.head_dim)
            self.k_norm, self.v_norm = nn.LayerNorm(self.head_dim), nn.LayerNorm(self.head_dim)

        if conv_kernel_size is not None and conv_kernel_size > 0:
            self.conv = nn.Conv2d(
                in_channels=self.num_heads,
                out_channels=self.num_heads,
                kernel_size=(conv_kernel_size, 1),
                padding=(conv_kernel_size // 2, 0),
                groups=self.num_heads
            )
            self.drop = nn.Dropout(self.dropout)
        else:
            self.conv, self.drop = None, None
        if activation_fn is None:
            self.activation_fn = functools.partial(F.softmax, dim=-1) 
        elif activation_fn == 'softmax':
            self.activation_fn = functools.partial(F.softmax, dim=-1)  
        elif activation_fn == 'sigmoid':
            self.activation_fn = F.sigmoid
        elif activation_fn == 'relu':
            self.activation_fn = F.relu
        elif activation_fn == 'identity':
            self.activation_fn = lambda x: x
        else:
            raise ValueError("Other activation functions cannot converge")
        
        # self.kv_norm = nn.LayerNorm(self.head_dim * self.head_dim)
        # self.qq_norm = nn.LayerNorm(self.head_dim * self.head_dim)
        # self.kk_norm = nn.LayerNorm(self.head_dim * self.head_dim)
        # self.qlatent_norm = nn.LayerNorm(self.num_landmarks)
        # self.qkv_norm = nn.LayerNorm(self.head_dim)
        self.out_norm = nn.LayerNorm(self.head_dim)
        
        # self.landmark_out_proj = nn.Linear(self.head_dim * 2, self.head_dim, bias=False)
        # nn.init.xavier_normal_(self.landmark_out_proj.weight, gain=2 ** -.5)
    
    def _create_temperature(self):
        temperature = nn.Parameter(torch.ones([self.num_heads, 1, 1]))
        nn.init.xavier_normal_(temperature, gain=2 ** -0.5)
        return temperature
        # nn.init.xavier_normal_(temperature, gain=2 ** -0.5)

    def _create_landmark(self):
        landmarks = nn.Parameter(torch.zeros((self.num_heads, self.ffn_dimension, self.head_dim)))
        nn.init.xavier_normal_(landmarks, gain=2 ** -.5)
        return landmarks

    def _apply_attention(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        bsz: int,
        attn_mask: Optional[Tensor] = None,
        query_padding_mask: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        r"""
        Computes scaled dot product attention on query, key and value tensors, using
        an optional attention mask if passed, and applying dropout if a probability
        greater than 0.0 is specified.
        Returns a tensor pair containing attended values and attention weights.
        Args:
            q, k, v: query, key and value tensors. See Shape section for shape details.
            attn_mask: optional tensor containing mask values to be added to calculated
                attention. May be 2D or 3D; see Shape section for details.
        Shape:
            - q: :math:`(B, Nt, E)` where B is batch size, Nt is the target sequence length,
                and E is embedding dimension.
            - key: :math:`(B, Ns, E)` where B is batch size, Ns is the source sequence length,
                and E is embedding dimension.
            - value: :math:`(B, Ns, E)` where B is batch size, Ns is the source sequence length,
                and E is embedding dimension.
            - attn_mask: either a 3D tensor of shape :math:`(B, Nt, Ns)` or a 2D tensor of
                shape :math:`(1, Nt, Ns)`.
            - Output: attention values have shape :math:`(B, Nt, E)`; attention weights
                have shape :math:`(B, Nt, wsize)`
        """
        # assert attn_mask is None, 'causal attention is not supported!'
        if self.add_norm:
            q, k, v = self.q_norm(q), self.k_norm(k), self.v_norm(v)
        # [bsz, d, c]
        weight_qk = self._calc_qk_weight(q, k, query_padding_mask, key_padding_mask)

        # print(torch.norm(latent_proj))
        # [bsz, c, d]
        weight_kv = self._calc_kv_weight(k, v, weight_qk, key_padding_mask)

        # print(torch.norm(kv_stat))
        # output = torch.softmax(q @ latent_proj, -1) @ kv_stat  
        output = self.activation_fn((q @ weight_qk) * (self.head_dim ** -0.5)) @ weight_kv  
        # output = F.normalize(q @ latent_proj, 2, -1) @ kv_stat  # nonlinear functions such as softmax, layer norm or tanh can be inserted into matmul
        # print(torch.norm(output))
        output = self.add_conv(output, q, query_padding_mask)
 
        # print(torch.norm(output))
        output = self.out_norm(output)

        # print(torch.norm(output))
        return output, None

    def add_conv(self, output, q, query_padding_mask):
        if self.conv is not None:
            q = q.reshape(q.shape[0] // self.num_heads, self.num_heads, q.shape[1], q.shape[2])
            if query_padding_mask is not None:
                q = q.masked_fill(query_padding_mask[:, None, :, None].to(torch.bool), 0.)
            conv_out = self.conv(q)
            conv_out = conv_out.reshape(q.shape[0] * self.num_heads, q.shape[2], q.shape[3])
            conv_out = F.relu(conv_out)
            conv_out = self.drop(conv_out)
            output = output + conv_out
        return output

    def _calc_qk_weight(self, q, k, q_padding_mask, k_padding_mask) -> Tensor:
        q = q.reshape(q.shape[0] // self.num_heads, self.num_heads, q.shape[1], self.head_dim)
        k = k.reshape(k.shape[0] // self.num_heads, self.num_heads, k.shape[1], self.head_dim)
        mus = [None, None]
        for i, (x, padding_mask, landmark, temperature) in enumerate(zip([q, k], 
                                                                         [q_padding_mask, k_padding_mask], 
                                                                         [self.q_landmarks, self.k_landmarks], 
                                                                         [self.qq_temperature, self.kk_temperature]
                                                                         )):
            # logits = torch.einsum(
            #     'bhnd,hcd->bhcn',
            #     x,
            #     landmark
            # )
            if padding_mask is not None:
                x = x.masked_fill(
                    padding_mask.unsqueeze(1).unsqueeze(-1).to(torch.bool),
                    0.0,
                )
            x = F.normalize(x, 2, 2)
            cov = x.transpose(2, 3) @ x * temperature 
            # cov = x_norm(cov.reshape(cov.shape[0], cov.shape[1], -1)).reshape(cov.shape[0], cov.shape[1], self.head_dim, self.head_dim)
            prob = F.softmax(cov, -1)
            feat = torch.einsum('bhdk,hcd->bhck', prob, landmark)
            # prob = F.softmax(logits, dim=-1)
            # feat = torch.einsum(
            #     'bhcn,bhnd->bhcd',
            #     prob,
            #     x
            # )
            mus[i] = feat
        # mus = torch.cat(mus, dim=-1).reshape(-1, self.ffn_dimension, self.head_dim * 2)
        # mu = self.landmark_out_proj(mus)
        mu = (mus[0] + mus[1]).reshape(-1, self.ffn_dimension, self.head_dim)
        if self.add_norm:
            mu = self.w_norm(mu)  # it can be softmax, layer norm or tanh along dimension c or d
        mu = F.relu(mu)
        return mu.transpose(-1, -2)

    def _calc_kv_weight(self, k: Tensor,
                        v: Tensor, 
                        landmarks: Tensor,
                        k_padding_mask: Tensor):


        if k_padding_mask is not None:  # different activation function requires different activations
            k = k.reshape(k.shape[0] // self.num_heads, self.num_heads, -1, self.head_dim)
            v = v.reshape(v.shape[0] // self.num_heads, self.num_heads, -1, self.head_dim)
            k = k.masked_fill(
                k_padding_mask.unsqueeze(1).unsqueeze(-1).to(torch.bool),
                0.0
            )
            v = v.masked_fill(
                k_padding_mask.unsqueeze(1).unsqueeze(-1).to(torch.bool),
                0.0
            )
        if len(k.shape) == 3:
            k = k.reshape(k.shape[0] // self.num_heads, self.num_heads, -1, self.head_dim)
            v = v.reshape(v.shape[0] // self.num_heads, self.num_heads, -1, self.head_dim)
        

        k = F.normalize(k, 2, 2)
        v = F.normalize(v, 2, 2)

        cov_kv = torch.einsum("bhnk,bhnl->bhkl", k, v) * self.kv_temperature
        prob = F.softmax(cov_kv, -1)
        prob = prob.reshape(prob.shape[0] * self.num_heads, -1, self.head_dim)
        kv_stat = prob @ landmarks

        return kv_stat.transpose(-1, -2)


class EMA(nn.Module):
    def __init__(self, momentum):
        super(EMA, self).__init__()
        self.momentum = momentum
        self.register_buffer('ema', torch.zeros(1))
        
    def forward(self, x: torch.Tensor):
        ema = x.mean(dim=1).mean(dim=0)
        self.ema = self.momentum * self.ema + (1 - self.momentum) * ema
        return x - self.ema



class AMLPQuery(MultiheadAttention):
    
    def __init__(
            self,
            *args,
            ffn_dimension=16,
            conv_kernel_size=None,
            activation_fn=None,
            add_norm=False,
            scale=False,
            add_ema=None,
            **kwargs):
        super(AMLPQuery, self).__init__(*args, **kwargs)
      
        self._reset_parameters()
        
        self.ffn_dimension = ffn_dimension
        self.add_norm = add_norm
        if self.add_norm:
            self.q_norm = nn.LayerNorm(self.head_dim)
            self.k_norm = nn.LayerNorm(self.head_dim)
            self.v_norm = nn.LayerNorm(self.head_dim)
            self.w_norm = nn.LayerNorm(self.head_dim)
        self.q_approx = self._create_approx()
        self.k_approx = self._create_approx()
        self.approx_out_proj = nn.Linear(2 * self.head_dim, self.head_dim)
        # self.kv_temperature = self._create_temperature()
        # self.qq_temperature = self._create_temperature()
        # self.kk_temperature = self._create_temperature()

        if add_ema is not None:
            self.add_ema = True 
            self.beta = add_ema
            self.ema_module = EMA(self.beta)
        else:
            self.add_ema = False
            
        self.scale = self.head_dim ** -0.5 if scale else 1
        if conv_kernel_size is not None and conv_kernel_size > 0:
            self.conv = nn.Conv2d(
                in_channels=self.num_heads,
                out_channels=self.num_heads,
                kernel_size=(conv_kernel_size, 1),
                padding=(conv_kernel_size // 2, 0),
                groups=self.num_heads
            )
            self.drop = nn.Dropout(self.dropout)
        else:
            self.conv, self.drop = None, None
        if activation_fn is None:
            self.activation_fn = functools.partial(F.softmax, dim=-1) 
        elif activation_fn == 'softmax':
            self.activation_fn = functools.partial(F.softmax, dim=-1)  
        elif activation_fn == 'sigmoid':
            self.activation_fn = F.sigmoid
        elif activation_fn == 'relu':
            self.activation_fn = F.relu
        elif activation_fn == 'identity':
            self.activation_fn = lambda x: x
        else:
            raise ValueError("Other activation functions cannot converge")

        self.out_norm = nn.LayerNorm(self.head_dim)

    def add_conv(self, output, q, query_padding_mask):
        if self.conv is not None:
            q = q.reshape(q.shape[0] // self.num_heads, self.num_heads, q.shape[1], q.shape[2])
            if query_padding_mask is not None:
                q = q.masked_fill(query_padding_mask[:, None, :, None].to(torch.bool), 0.)
            conv_out = self.conv(q)
            conv_out = conv_out.reshape(q.shape[0] * self.num_heads, q.shape[2], q.shape[3])
            conv_out = F.relu(conv_out)
            conv_out = self.drop(conv_out)
            output = output + conv_out
        return output
    def _create_temperature(self):
        temperature = nn.Parameter(torch.ones([self.num_heads, 1, 1]))
        nn.init.xavier_normal_(temperature, gain=2 ** -0.5)
        return temperature


    def _create_approx(self):
        landmarks = nn.Parameter(torch.zeros((self.num_heads, self.ffn_dimension, self.head_dim)))
        nn.init.xavier_normal_(landmarks, gain=2 ** -.5)
        return landmarks

    def _apply_attention(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        bsz: int,
        attn_mask: Optional[Tensor] = None,
        query_padding_mask: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        if attn_mask is not None:
            warnings.warn("AMLP does not support causal attention")
        # assert attn_mask is None, 'causal attention is not supported!'
        if self.add_norm:
            q, k, v = self.q_norm(q), self.k_norm(k), self.v_norm(v)
        if self.add_ema:
            # temp_cumsum = torch.cumsum(q, -2)[:, 1:]  # b, n, d
            # total = torch.arange(1, q.shape[-2]).to(q)[None, :, None].repeat(bsz * self.num_heads, 1, self.head_dim)
            # # print(temp_cumsum.shape, total.shape)
            # temp_cummean = torch.cat([torch.zeros((bsz * self.num_heads, 1, self.head_dim)).to(q), temp_cumsum / total], dim=-2)
            # q = q * self.beta + (1 - self.beta) * temp_cummean
            q = self.ema_module(q)
        weight_qk = self._calc_qk_weight(q, k, query_padding_mask, key_padding_mask)

        weight_kv = self._calc_kv_weight(k, v, weight_qk, key_padding_mask)

        output = self.activation_fn(q @ weight_qk * self.scale) @ weight_kv  
    
        output = self.add_conv(output, q, query_padding_mask)

        output = self.out_norm(output)

        return output, None

    def _calc_qk_weight(self, q, k, q_padding_mask, k_padding_mask) -> Tensor:
        q = q.reshape(q.shape[0] // self.num_heads, self.num_heads, q.shape[1], self.head_dim)
        k = k.reshape(k.shape[0] // self.num_heads, self.num_heads, k.shape[1], self.head_dim)
        mus = [None, None]
        for i, (x, padding_mask, approx) in enumerate(zip([q, k], [q_padding_mask, k_padding_mask], [self.q_approx, self.k_approx])):
            logits = torch.einsum(
                'bhnd,hcd->bhcn',
                x,
                approx
            )
            if padding_mask is not None:
                logits = logits.masked_fill(
                    padding_mask.unsqueeze(1).unsqueeze(1).to(torch.bool),
                    float('-inf'),
                )
            prob = F.softmax(logits, dim=-1)
            feat = torch.einsum(
                'bhcn,bhnd->bhcd',
                prob,
                x
            )
            mus[i] = feat
        mus = torch.cat(mus, dim=-1).reshape(-1, self.ffn_dimension, self.head_dim * 2)
        mu = self.approx_out_proj(mus)
        if self.add_norm:
            mu = self.w_norm(mu)  # it can be softmax, layer norm or tanh along dimension c or d
        return mu.transpose(-1, -2)

    def _calc_kv_weight(self, k, v, approx, k_padding_mask):
        logits = torch.einsum(
            'bmd,bdc->bcm',
            k,
            approx
        )
        if k_padding_mask is not None:  # different activation function requires different activations
            logits = logits.view(logits.shape[0] // self.num_heads, self.num_heads, self.ffn_dimension, logits.shape[2])
            logits = logits.masked_fill(
                k_padding_mask.unsqueeze(1).unsqueeze(1).to(torch.bool),
                float('-inf')
            )
            logits = logits.view(logits.shape[0] * self.num_heads, self.ffn_dimension, logits.shape[3])
        prob = F.softmax(logits, dim=-1)  # softmax could be replaced with other nonlinear activations
        kv_stat = prob @ v
        return kv_stat




def _prep_mask(
    attn_mask: Optional[Tensor],
    key_padding_mask: Optional[Tensor]
) -> Tuple[Optional[Tensor], Optional[Tensor]]:
    if attn_mask is not None:
        if attn_mask.dtype == torch.uint8:
            warnings.warn("Byte tensor for attn_mask in nn.MultiheadAttention is deprecated. "
                          "Use bool tensor instead.")
            attn_mask = attn_mask.to(torch.bool)
        if attn_mask.dim() == 2:
            attn_mask = attn_mask.unsqueeze(0)

    if key_padding_mask is not None:
        if key_padding_mask.dtype == torch.uint8:
            warnings.warn("Byte tensor for key_padding_mask in nn.MultiheadAttention is deprecated. "
                          "Use bool tensor instead.")
            key_padding_mask = key_padding_mask.to(torch.bool)
    return attn_mask, key_padding_mask

