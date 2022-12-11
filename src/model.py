import torch
import torch.nn as nn
from transformer.layers import TransformerLayer, Decoder_layer, PositionalEmbedding
from utils import NOTE_LEVEL_DICT


class NoteTransformer(nn.Module):
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
        super(NoteTransformer, self).__init__()
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
        # audio encoder
        self.ln_src_enc = nn.LayerNorm(d_src_enc)
        self.enc_emb = nn.Linear(d_src_enc, d_emb_enc)
        self.position_emb_enc = PositionalEmbedding(d_emb_enc, n_position=max_len_enc)
        self.ln_enc_1 = nn.LayerNorm(d_emb_enc)

        # decoder
        self.dec_emb = nn.Embedding(
            n_labels, d_emb_dec, padding_idx=NOTE_LEVEL_DICT["<pad>"]
        )
        self.position_emb_dec = PositionalEmbedding(d_emb_dec, n_position=max_len_dec)
        self.ln_dec = nn.LayerNorm(d_emb_dec)

    def embedding(self, **kwargs):
        enc_emb = self.enc_embedding(**kwargs)
        dec_emb = self.dec_embedding(**kwargs)
        return enc_emb, dec_emb

    def enc_embedding(self, **kwargs):
        # audio encoder
        enc_emb = self.enc_emb(self.ln_src_enc(kwargs["spec_origin"]))
        enc_emb += self.position_emb_enc(enc_emb)
        enc_emb = self.ln_enc_1(enc_emb)
        return enc_emb

    def dec_embedding(self, **kwargs):
        dec_emb = self.dec_emb(kwargs["dec_seq"])
        dec_emb += self.position_emb_dec(dec_emb)
        dec_emb = self.ln_dec(dec_emb)

        return dec_emb

    def _encoding(self, enc_out):

        for enc_layer in self.encoder:
            enc_out, _ = enc_layer(enc_out)

        return enc_out

    def _decoding(self, enc_out, dec_out):

        for dec_layer in self.decoder:
            dec_out, _ = dec_layer(enc_out, dec_out)

        logits = self.trg_prj(dec_out)

        return dec_out, logits

    def forward(self, **kwargs):

        enc_emb, dec_emb = self.embedding(**kwargs)

        enc_out = self._encoding(enc_emb)

        dec_out, logits = self._decoding(enc_out, dec_emb)

        return {"logits": logits}

    def _make_masks(self, device):
        if not hasattr(self, "masks"):

            time_mask = torch.zeros(self.n_labels).to(device) == 1
            pitch_mask = torch.zeros(self.n_labels).to(device) == 1
            for k, v in NOTE_LEVEL_DICT.items():
                if "time" in k:
                    pitch_mask[v] = True
                if "pitch" in k:
                    time_mask[v] = True
                if "offset" in k:
                    time_mask[v] = True
                if "eos" in k:
                    pitch_mask[v] = True
            self.masks = [time_mask, pitch_mask]

    def decoding(self, **kwargs):

        num_time_idx = sum(["time" in i for i in NOTE_LEVEL_DICT])
        max_len_dec = self.position_emb_dec.pos_table.shape[1]

        spec_seq = kwargs["spec_origin"]

        # prime token sequence existence
        if "dec_seq" not in kwargs:
            kwargs["dec_seq"] = (
                torch.LongTensor([NOTE_LEVEL_DICT["<sos>"]] * spec_seq.shape[0])
                .to(spec_seq.device)
                .unsqueeze(-1)
            )

        # sequential rule mask
        self._make_masks(spec_seq.device)

        # padding after eos token
        padding = torch.zeros(spec_seq.shape[0]).to(spec_seq.device).unsqueeze(-1) == 1

        # compute encoder output
        enc_emb = self.enc_embedding(**kwargs)
        enc_out = self._encoding(enc_emb)

        # autoregressive decoding
        while not torch.all(
            (kwargs["dec_seq"] == NOTE_LEVEL_DICT["<eos>"]).sum(1) >= 1
        ):
            dec_emb = self.dec_embedding(**kwargs)
            dec_out, outputs = self._decoding(enc_out, dec_emb)

            # sequential rule mask
            mask = self.masks[(outputs.shape[1] - 1) % 2].clone()

            # overlap decoding case (batch size 1) and order of time prediction
            if (
                kwargs["dec_seq"].shape[0] == 1
                and ((kwargs["dec_seq"].shape[1] - 1) % 2) == 0
            ):
                # time index should increase monotonically
                if kwargs["dec_seq"].shape[1] >= 3:
                    mask[
                        : torch.max(
                            kwargs["dec_seq"][kwargs["dec_seq"] <= num_time_idx]
                        )
                    ] = True

                indices, counts = torch.unique(
                    kwargs["dec_seq"], return_counts=True, dim=-1
                )
                # if same time token occurs more than 3 times, apply mask the time token
                if ((indices < num_time_idx) * (counts >= 3)).sum():
                    for idx in indices[(indices <= num_time_idx) * (counts >= 3)]:
                        mask[: idx + 1] = True

            outputs[:, -1, mask] = -float("Inf")

            # argmax
            predictions = outputs[:, -1, :].argmax(-1, keepdim=True)

            # force stop for last step
            if kwargs["dec_seq"].shape[1] == max_len_dec - 1:
                predictions[:] = NOTE_LEVEL_DICT["<eos>"]

            # padding after eos token
            predictions[padding] = NOTE_LEVEL_DICT["<pad>"]
            kwargs["dec_seq"] = torch.cat((kwargs["dec_seq"], predictions), dim=-1)
            padding += predictions == NOTE_LEVEL_DICT["<eos>"]

        return kwargs["dec_seq"]
