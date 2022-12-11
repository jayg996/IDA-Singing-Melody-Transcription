from torch import nn
from .layers import TransformerLayer, Decoder_layer, PositionalEmbedding


class Transformer(nn.Module):
    def __init__(
        self,
        d_src_enc,
        d_emb_enc,
        max_len_enc,
        n_layers_enc,
        n_head_enc,
        d_k_enc,
        d_v_enc,
        d_inner_enc,
        d_emb_dec,
        max_len_dec,
        n_layers_dec,
        n_head_dec,
        d_k_dec,
        d_v_dec,
        d_inner_dec,
        dropout,
        attn_dropout,
        n_labels,
        conv_kernel=1,
        use_rel_pos=False,
    ):
        super(Transformer, self).__init__()
        self.use_rel_pos = use_rel_pos
        self.n_labels = n_labels
        self.init_emb_layers(
            d_src_enc, d_emb_enc, max_len_enc, n_labels, d_emb_dec, max_len_dec
        )

        self.encoder = nn.ModuleList(
            [
                TransformerLayer(
                    d_emb_enc,
                    d_inner_enc,
                    n_head_enc,
                    d_k_enc,
                    d_v_enc,
                    conv_kernel,
                    max_len_enc,
                    dropout,
                    attn_dropout,
                    causal=False,
                    use_rel_pos=use_rel_pos,
                )
                for _ in range(n_layers_enc)
            ]
        )

        self.decoder = nn.ModuleList(
            [
                Decoder_layer(
                    d_emb_dec,
                    d_inner_dec,
                    n_head_dec,
                    d_k_dec,
                    d_v_dec,
                    conv_kernel,
                    max_len_dec,
                    dropout,
                    attn_dropout,
                    causal=True,
                    use_rel_pos=use_rel_pos,
                )
                for _ in range(n_layers_dec)
            ]
        )

        self.trg_prj = nn.Linear(d_emb_dec, n_labels)

    def init_emb_layers(
        self, d_src_enc, d_emb_enc, max_len_enc, n_labels, d_emb_dec, max_len_dec
    ):
        # encoder
        self.ln_src_enc = nn.LayerNorm(d_src_enc)
        self.enc_emb = nn.Linear(d_src_enc, d_emb_enc)
        self.position_emb_enc = PositionalEmbedding(d_emb_enc, n_position=max_len_enc)
        self.ln_enc = nn.LayerNorm(d_emb_enc)

        # decoder
        self.dec_emb = nn.Embedding(n_labels, d_emb_dec)
        self.position_emb_dec = PositionalEmbedding(d_emb_dec, n_position=max_len_dec)
        self.ln_dec = nn.LayerNorm(d_emb_dec)

    def embedding(self, enc_seq, dec_seq):
        # encoder
        enc_emb = self.enc_emb(self.ln_src_enc(enc_seq))
        enc_emb += self.position_emb_enc(enc_emb)
        enc_emb = self.ln_enc(enc_emb)

        # decoder
        dec_emb = self.dec_emb(dec_seq)
        dec_emb += self.position_emb_dec(dec_emb)
        dec_emb = self.ln_dec(dec_emb)

        return enc_emb, dec_emb

    def forward(self, enc_seq, dec_seq):
        enc_out, dec_out = self.embedding(enc_seq, dec_seq)

        for enc_layer in self.encoder:
            enc_out, _ = enc_layer(enc_out)

        for dec_layer in self.decoder:
            dec_out, _ = dec_layer(enc_out, dec_out)

        logits = self.trg_prj(dec_out)
        return {"logits": logits}
