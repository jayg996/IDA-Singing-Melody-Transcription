import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


def get_pad_mask(seq, pad_idx):
    return (seq != pad_idx).unsqueeze(-2)


def get_causal_mask(seq):
    """
    Generates bias values (-Inf) to mask future timesteps during attention
    """
    len_s = seq.size(2)
    np_mask = np.triu(np.full([len_s, len_s], -np.inf), 1)
    torch_mask = torch.from_numpy(np_mask).to(device=seq.device, dtype=torch.float32)
    return torch_mask.unsqueeze(0)


class RelativePositionalEmbedding(nn.Module):
    def __init__(self, d_k, num_heads, max_len, causal=False):
        super(RelativePositionalEmbedding, self).__init__()

        self.causal = causal
        self.max_len = max_len
        rel_emb_len = max_len if causal else 2 * max_len - 1
        rel_emb = nn.Parameter(torch.randn(num_heads, d_k, rel_emb_len))
        self.register_parameter("rel_emb", rel_emb)

    def forward(self, queries):
        # [batch_size, num_heads, seq_length, depth/num_heads]
        # causal autoregressive, 아니면 -l+1 ~ l-1 짜리 relative positional embedding
        embedding = torch.matmul(queries, self.rel_emb)  # b h l l or b h l 2l-1
        if self.causal:
            embedding = self._qe_masking(embedding)
            embedding = F.pad(embedding, (1, 0, 0, 0))  # b h l l+1
            embedding = embedding.view(
                -1, embedding.size(1), embedding.size(3), embedding.size(2)
            )  # b h l+1 l
            embedding = embedding[:, :, 1:, :]  # b h l l
        else:
            embedding = F.pad(embedding, (1, 0, 0, 0))  # b h l 2l
            embedding = embedding.view(embedding.size(0), embedding.size(1), -1)[
                :, :, self.max_len :
            ]
            embedding = embedding.view(
                embedding.size(0), embedding.size(1), self.max_len, -1
            )[:, :, :, : self.max_len]

        return embedding  # batch_size * num_heads * max_len * max_len

    @staticmethod
    def _qe_masking(qe):
        lengths = torch.arange(qe.size(-1) - 1, qe.size(-1) - qe.size(-2) - 1, -1)
        maxlen = qe.size(-1)
        mask = torch.arange(maxlen).unsqueeze(0) >= lengths.unsqueeze(1)
        return mask.float().to(qe.device) * qe


class PositionalEmbedding(nn.Module):
    def __init__(self, d_hid, n_position=200):
        super(PositionalEmbedding, self).__init__()

        # Not a parameter
        self.register_buffer(
            "pos_table", self._get_sinusoid_encoding_table(n_position, d_hid)
        )

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        """Sinusoid position encoding table"""
        # TODO: make it with torch instead of numpy

        def get_position_angle_vec(position):
            return [
                position / np.power(10000, 2 * (hid_j // 2) / d_hid)
                for hid_j in range(d_hid)
            ]

        sinusoid_table = np.array(
            [get_position_angle_vec(pos_i) for pos_i in range(n_position)]
        )
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        return self.pos_table[:, : x.size(1)].clone().detach()


class ScaledDotProductAttention(nn.Module):
    """Scaled Dot-Product Attention"""

    def __init__(self, temperature, attn_dropout=0.0, rel_pos_emb=None):
        super(ScaledDotProductAttention, self).__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.rel_pos_emb = rel_pos_emb

    def forward(self, q, k, v, mask):
        # batch_size * num heads * seq_len * depth --> batch_size * num heads * seq_len * seq_len
        attn = torch.matmul(q, k.transpose(2, 3)) / self.temperature

        # use relative positional embedding
        if self.rel_pos_emb:
            rel_emb = self.rel_pos_emb(q) / self.temperature
            # query, key length adaptation
            # rel_emb = rel_emb[:, :, :attn.shape[2], :attn.shape[3]]
            attn = attn + rel_emb

        # masking
        if mask is not None:
            attn += mask

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention module"""

    def __init__(
        self,
        n_head,
        d_model,
        d_k,
        d_v,
        max_len,
        dropout=0.1,
        attn_dropout=0.0,
        causal=False,
        use_rel_pos=False,
    ):
        super(MultiHeadAttention, self).__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.max_len = max_len
        self.causal = causal

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        # self attention module
        rel_pos_emb = (
            RelativePositionalEmbedding(d_k, n_head, max_len, causal)
            if use_rel_pos
            else None
        )
        self.attention = ScaledDotProductAttention(
            temperature=d_k ** 0.5, attn_dropout=attn_dropout, rel_pos_emb=rel_pos_emb
        )

        self.dropout = nn.Dropout(dropout)
        self.layer_norm_bff = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)
        residual = q

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(
            sz_b, len_q, n_head, d_k
        )  # batch, query time len, num heads, dimension
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if self.causal:
            mask = get_causal_mask(k)
            mask = mask.unsqueeze(1)  # For head axis broadcasting.

        q, attn = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lenq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        q += residual
        q = self.layer_norm_bff(q)

        return q, attn


class PositionwiseConv(nn.Module):
    """A two-1D-convolutional-layer module"""

    def __init__(self, d_in, d_hid, conv_kernel=1, dropout=0.1):
        super(PositionwiseConv, self).__init__()
        if conv_kernel % 2 != 1:
            raise NotImplementedError
        padding = conv_kernel // 2
        # conv kernel 사이즈가 1인경우 timewise fc랑 같음
        self.w_1 = nn.Conv1d(
            d_in, d_hid, kernel_size=conv_kernel, padding=padding
        )  # position-wise
        self.w_2 = nn.Conv1d(
            d_hid, d_in, kernel_size=conv_kernel, padding=padding
        )  # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: batch size * seq_len * dimension --> batch size * dimension * seq_len
        x = x.permute(0, 2, 1)
        residual = x
        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual
        x = x.permute(0, 2, 1)
        x = self.layer_norm(x)

        return x


class TransformerLayer(nn.Module):
    """Compose with two layers"""

    def __init__(
        self,
        d_model,
        d_inner,
        n_head,
        d_k,
        d_v,
        conv_kernel,
        max_len,
        dropout=0.1,
        attn_dropout=0.0,
        causal=False,
        use_rel_pos=False,
    ):
        super(TransformerLayer, self).__init__()
        self.causal = causal
        self.slf_attn = MultiHeadAttention(
            n_head,
            d_model,
            d_k,
            d_v,
            max_len,
            dropout=dropout,
            attn_dropout=attn_dropout,
            causal=causal,
            use_rel_pos=use_rel_pos,
        )
        self.pos_conv = PositionwiseConv(d_model, d_inner, conv_kernel, dropout=dropout)

    def forward(self, enc_input):
        enc_output, enc_slf_attn = self.slf_attn(enc_input, enc_input, enc_input)
        enc_output = self.pos_conv(enc_output)
        return enc_output, enc_slf_attn


class Decoder_layer(TransformerLayer):
    def __init__(
        self,
        d_model,
        d_inner,
        n_head,
        d_k,
        d_v,
        conv_kernel,
        max_len,
        dropout=0.1,
        attn_dropout=0.0,
        causal=False,
        use_rel_pos=False,
    ):
        super(Decoder_layer, self).__init__(
            d_model,
            d_inner,
            n_head,
            d_k,
            d_v,
            conv_kernel,
            max_len,
            dropout,
            attn_dropout,
            causal,
            use_rel_pos,
        )

        self.enc_dec_attn = MultiHeadAttention(
            n_head,
            d_model,
            d_k,
            d_v,
            max_len,
            dropout=dropout,
            attn_dropout=attn_dropout,
            causal=False,
            use_rel_pos=use_rel_pos,
        )

    def forward(self, enc_output, dec_input, mask=None):
        dec_output, slf_attn_probs = self.slf_attn(dec_input, dec_input, dec_input)
        dec_output, enc_dec_attn_probs = self.enc_dec_attn(
            dec_output, enc_output, enc_output, mask
        )
        dec_output = self.pos_conv(dec_output)
        return dec_output, [slf_attn_probs, enc_dec_attn_probs]
