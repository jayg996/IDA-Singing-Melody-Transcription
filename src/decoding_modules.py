import os
import torch
import math
import pretty_midi as pm
from utils import NOTE_LEVEL_DICT, NOTE_LEVEL_CLASS
from audio_processor import AudioProcessor
from stft import STFT

audio_processor = AudioProcessor()
signal_processor = STFT(filter_length=2048, hop_length=160)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_encoder_inputs(audio_path, sr):

    # spectrogram
    waveform, _ = audio_processor.load(audio_path, sr=sr)
    waveform = waveform.to(device)
    spectrogram, _ = signal_processor(waveform)
    spectrogram = spectrogram.squeeze(0).permute(1, 0)
    total_len = len(spectrogram)

    return spectrogram, total_len


def decoding(
    model,
    spectrogram,
    total_len,
    win_size,
    unit_time,
    decoding_overlap=True,
    decoding_hop_size=256,
    remove_end_size=128,
):

    # overlap
    if decoding_overlap:

        # padding
        inputs = dict()
        len_pad = (
            math.ceil((total_len - win_size) / decoding_hop_size) * decoding_hop_size
            + win_size
        ) - total_len

        padding_feature = torch.zeros((len_pad, spectrogram.shape[1])).to(device)
        inputs["spec_origin"] = torch.cat((spectrogram, padding_feature), dim=0)

        # overlapping decoding
        outputs = list()
        next_prime = list()
        position = 0
        while position + win_size <= (total_len + len_pad):
            tmp_inputs = dict()
            for k, v in inputs.items():
                tmp_inputs[k] = v[position : position + win_size].unsqueeze(0)

            tmp_inputs["dec_seq"] = (
                torch.LongTensor([NOTE_LEVEL_DICT["<sos>"]] * 1)
                .to(device)
                .unsqueeze(-1)
            )
            if len(next_prime):
                tmp_inputs["dec_seq"] = torch.cat(
                    (tmp_inputs["dec_seq"], next_prime), dim=1
                )

            dec_seq = model.decoding(**tmp_inputs)
            dec_seq = dec_seq[0][1:-1]
            assert len(dec_seq) % 2 == 0, "2 tokens per event"
            dec_seq = dec_seq.reshape(-1, 2)
            save_seq = list()
            next_prime = list()
            for i in dec_seq:
                time = float(NOTE_LEVEL_CLASS[i[0]][6:-1])
                next_start = unit_time * decoding_hop_size
                remove_end = unit_time * (win_size - remove_end_size)

                # in last segment, save all
                if (position + win_size) == (total_len + len_pad):
                    save_seq.append(i.clone())
                    continue

                # save sequence
                if time < next_start:
                    save_seq.append(i.clone())
                # remove end sequence
                elif time > remove_end:
                    continue
                # next prime sequence
                else:
                    new_time = time - next_start
                    tmp_event = i.clone()
                    tmp_event[0] = NOTE_LEVEL_DICT[f"<time {new_time:0.2f}>"]
                    next_prime.append(tmp_event)

            if len(next_prime):
                next_prime = torch.stack(next_prime).reshape(1, -1)
            save_seq = (
                [torch.LongTensor([NOTE_LEVEL_DICT["<sos>"]]).to(device)]
                + save_seq
                + [torch.LongTensor([NOTE_LEVEL_DICT["<eos>"]]).to(device)]
            )
            save_seq = torch.cat(save_seq).reshape(1, -1)
            outputs += [s for s in save_seq]
            position += decoding_hop_size

    # non overlap
    else:
        batch_size = 8
        inputs = dict()

        # padding
        padNum = total_len % win_size
        len_pad = win_size - padNum

        padding_feature = torch.zeros((len_pad, spectrogram.shape[1])).to(device)
        inputs["spec_origin"] = torch.cat((spectrogram, padding_feature), dim=0)
        inputs["spec_origin"] = inputs["spec_origin"].reshape(
            -1, win_size, inputs["spec_origin"].shape[1]
        )

        # batch decoding
        outputs = list()
        for i in range(math.ceil(inputs["spec_origin"].shape[0] / batch_size)):
            tmp_inputs = dict()
            for k, v in inputs.items():
                tmp_inputs[k] = v[
                    batch_size
                    * i : min(batch_size * (i + 1), inputs["spec_origin"].shape[0])
                ]

            dec_seq = model.decoding(**tmp_inputs)
            outputs += [s for s in dec_seq]
        decoding_hop_size = win_size

    return outputs, decoding_hop_size


def token_sequence_to_notes(outputs, hop_size, unit_time):

    note_events = list()
    for i in range(len(outputs)):
        base_time = (i * hop_size) * unit_time
        j = 0
        while True:
            if outputs[i][j] == NOTE_LEVEL_DICT["<sos>"]:
                j += 1
                pass
            elif outputs[i][j] == NOTE_LEVEL_DICT["<eos>"]:
                break
            elif outputs[i][j] < 1024:
                time = float(NOTE_LEVEL_CLASS[outputs[i][j]][6:-1]) + base_time
                if outputs[i][j + 1] == NOTE_LEVEL_DICT["<offset>"]:
                    pitch = "<offset>"
                else:
                    pitch = int(NOTE_LEVEL_CLASS[outputs[i][j + 1]][7:-1])
                note_events.append([time, pitch])
                j += 2

    # save monophonic melody
    notes = list()
    tmp_note = dict()

    for t, p in note_events:
        if p != "<offset>":
            if len(tmp_note) != 0:
                notes.append([tmp_note["start"], t, tmp_note["pitch"]])
                tmp_note = dict()
                print("onset without offset")
            tmp_note["start"] = t
            tmp_note["pitch"] = p
        else:
            if len(tmp_note) == 0:
                print("offset without onset")
                pass
            else:
                notes.append([tmp_note["start"], t, tmp_note["pitch"]])
                tmp_note = dict()
    return notes


def audio_to_result(
    model,
    audio_path,
    save_dir,
    sr=16000,
    hop_length=160,
    crop_length=160 * 512,
):
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(
        save_dir, os.path.splitext(os.path.basename(audio_path))[0] + ".lab"
    )
    if os.path.exists(save_path):
        return save_path

    # data config
    fps = sr / hop_length
    unit_time = 1 / fps
    win_size = crop_length // hop_length + 1

    # load encoder inputs
    label_list, total_len = load_encoder_inputs(audio_path, sr)

    # decoding
    outputs, hop_size = decoding(model, label_list, total_len, win_size, unit_time)

    # token sequence to notes
    notes = token_sequence_to_notes(outputs, hop_size, unit_time)

    # notes to midi
    midi = pm.PrettyMIDI()
    instrument = pm.Instrument(program=0)
    lines = []
    for i in notes:
        # avoid duration zero note
        if i[0] != i[1]:
            lines.append("%.2f %.2f %d\n" % (i[0], i[1], i[2]))
            pm_note = pm.Note(
                velocity=120, pitch=i[2], start=round(i[0], 2), end=round(i[1], 2)
            )
            instrument.notes.append(pm_note)

    # save lab file
    with open(save_path, "w") as f:
        for line in lines:
            f.write(line)

    # save midi
    midi.instruments.append(instrument)
    midi.write(save_path.replace(".lab", ".midi"))
