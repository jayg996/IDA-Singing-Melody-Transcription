import torch
from model import NoteTransformer
from decoding_modules import audio_to_result

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# hyperparameters
model_config = {
    "d_src_enc": 1025,
    "d_emb_enc": 512,
    "max_len_enc": 513,
    "n_layers_enc": 8,
    "n_head_enc": 8,
    "d_k_enc": 64,
    "d_v_enc": 64,
    "d_inner_enc": 1024,
    "d_emb_dec": 512,
    "max_len_dec": 512,
    "n_layers_dec": 8,
    "n_head_dec": 8,
    "d_k_dec": 64,
    "d_v_dec": 64,
    "d_inner_dec": 1024,
    "dropout": 0.1,
    "attn_dropout": 0.1,
    "n_labels": 1157,
}

# load model
model = NoteTransformer(**model_config)
model = model.to(device)
model.eval()

# load parameters
asset_path = "transformer+OD+PA+AD.pth.tar"
checkpoint = torch.load(asset_path, map_location=device)
model.load_state_dict(checkpoint["model"])

# inference file
mp3_path = "test.mp3"
save_dir = "save"

# singing melody transcription
audio_to_result(model, mp3_path, save_dir)
