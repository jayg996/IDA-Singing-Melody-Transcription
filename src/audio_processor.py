import librosa
import torch
import torchaudio
import numpy as np
from madmom.audio.signal import Signal
from pysndfx import AudioEffectsChain


class AudioProcessor:
    def __init__(self, device=None):
        self.device = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if device is None
            else device
        )

    def load(
        self,
        path: str,
        sr: int = None,
        mono: bool = True,
        backend: str = "madmom",
        norm: bool = True,
    ):
        """[summary]

        Args:
            path (str): audio file path (wav, mp3, ...).
            sr (int, optional): sampling rate. Defaults to None.
            mono (bool, optional): convert signal to mono. Defaults to True.
            backend (str, optional): Name of the backend. Defaults to "madmom".
            norm (bool, optional): Normalize the signal to [-1,+1]. Defaults to True.

        Returns:
            waveform (torch.FloatTensor): (C, T) where C = number of channels, T = number of samples
            sr (int): sampling rate
        """

        if backend == "madmom":
            num_channels = 1 if mono else None
            sig = Signal(
                path,
                sample_rate=sr,
                dtype=np.float32,
                num_channels=num_channels,
                norm=norm,
            )
            waveform = torch.from_numpy(np.array(sig))
            if sr is None:
                sr = sig.sample_rate
            if len(waveform.shape) == 1:
                waveform = waveform.unsqueeze(0)
            elif len(waveform.shape) == 2:
                waveform = waveform.permute(1, 0)

        elif backend == "librosa":
            waveform, sr = librosa.load(path, sr=sr, mono=mono)
            waveform = torch.FloatTensor(waveform)
            if norm:
                scale = waveform.abs().max()
                waveform /= scale
            if len(waveform.shape) == 1:
                waveform = waveform.unsqueeze(0)

        elif backend == "torchaudio":
            waveform, sample_rate = torchaudio.load(path)
            if sr is not None:
                if sample_rate != sr:
                    resample = torchaudio.transforms.Resample(sample_rate, sr)
                    waveform = resample(waveform)
            else:
                sr = sample_rate
            if mono:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            if norm:
                scale = waveform.abs().max()
                waveform /= scale

        return waveform, sr

    def save(self, path: str, waveform: torch.FloatTensor, sr: int):
        """[summary]

        Args:
            path (str): audio file path (wav, mp3, ...).
            waveform (torch.FloatTensor): (T,) or (C, T) where C = number of channels, T = number of samples
            sr (int): sampling rate
        """

        if len(waveform.shape) == 1:
            waveform = waveform.unsqueeze(0)
        torchaudio.save(path, waveform.cpu(), sr)

    def pitch_shift(
        self, waveform: torch.FloatTensor, shift: float, semitone: bool = True
    ):
        """[summary]

        Args:
            waveform (torch.FloatTensor): (C, T) where C = number of channels, T = number of samples
            shift (float): pitch shift factor
            semitone (bool, optional): if True, shift is semitone scale. if False, cent scale. Defaults to True.

        Returns:
            waveform (torch.FloatTensor): (C, T) where C = number of channels, T = number of samples
        """
        if shift == 0:
            return waveform

        if semitone:
            shift *= 100

        results = []
        fx = AudioEffectsChain()
        fx.pitch(shift)
        for i in range(len(waveform)):
            result = torch.from_numpy(fx(waveform[i].numpy()))
            # remove difference of num samples
            if result.shape[-1] > waveform.shape[-1]:
                result = result[: waveform.shape[-1]]
            elif result.shape[-1] < waveform.shape[-1]:
                result = torch.nn.functional.pad(
                    result, (0, waveform.shape[-1] - result.shape[-1])
                )
            results.append(result)
        waveform = torch.stack(results)

        return waveform

    def tempo_shift(self, waveform: torch.FloatTensor, factor: float):
        """[summary]
        This effect changes the duration of the sound without modifying pitch.

        Args:
            waveform (torch.FloatTensor): (C, T) where C = number of channels, T = number of samples
            shift (float): tempo shift factor

        Returns:
            waveform (torch.FloatTensor): (C, T) where C = number of channels, T = number of samples
        """
        if factor == 1:
            return waveform

        results = []
        fx = AudioEffectsChain()
        fx.tempo(factor)
        for i in range(len(waveform)):
            result = torch.from_numpy(fx(waveform[i].numpy()))
            results.append(result)
        waveform = torch.stack(results)

        return waveform

    def speed_shift(self, waveform: torch.FloatTensor, factor: float):
        """[summary]
        A factor of 2 doubles the speed and raises the pitch an octave.

        Args:
            waveform (torch.FloatTensor): (C, T) where C = number of channels, T = number of samples
            shift (float): speed shift factor

        Returns:
            waveform (torch.FloatTensor): (C, T) where C = number of channels, T = number of samples
        """

        results = []
        fx = AudioEffectsChain()
        fx.speed(factor)
        for i in range(len(waveform)):
            result = torch.from_numpy(fx(waveform[i].numpy()))
            results.append(result)
        waveform = torch.stack(results)

        return waveform

    def gain(self, waveform: torch.FloatTensor, db: float):
        """[summary]

        Args:
            waveform (torch.FloatTensor): (C, T) where C = number of channels, T = number of samples
            db (float): gain decibel

        Returns:
            waveform (torch.FloatTensor): (C, T) where C = number of channels, T = number of samples
        """
        results = []
        fx = AudioEffectsChain()
        fx.gain(db)
        for i in range(len(waveform)):
            result = torch.from_numpy(fx(waveform[i].numpy()))
            results.append(result)
        waveform = torch.stack(results)

        return waveform
