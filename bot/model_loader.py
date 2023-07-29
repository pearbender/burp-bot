import torch
import torchaudio
from torchaudio import transforms
import numpy as np

from model import *


def prepare_file(audio_file: str) -> torch.Tensor:
    sig, sr = torchaudio.load(audio_file)
    spec = transforms.MelSpectrogram(
        sr, n_fft=1024, hop_length=None, n_mels=64)(sig)
    spec = transforms.AmplitudeToDB(top_db=80)(spec)

    return spec.unsqueeze(0)


def prepare_array(array: np.ndarray, sr: int) -> torch.Tensor:
    sig = torch.tensor(array)
    spec = transforms.MelSpectrogram(
        sr, n_fft=1024, hop_length=None, n_mels=64)(sig)
    spec = transforms.AmplitudeToDB(top_db=80)(spec)

    return spec.unsqueeze(0)


def load_model(model_filename: str):
    model = AudioClassifier()
    model.load_state_dict(torch.load(model_filename))
    model.eval()
    return model


class BurpEvaluator:
    models = []
    sr = 44100


    def __init__(self, model_files, sr=44100) -> None:
        self.models = [load_model(m) for m in model_files]
        self.sr = sr


    def _evaluate_with_model(self, model, tensor: torch.Tensor) -> bool:
        output = model.forward(tensor)
        _, classes = torch.max(output, 1)
        return classes.item() == 0


    def evaluate_file(self, filename) -> bool:
        tensor = prepare_file(filename)
        return self.evaluate_tensor(tensor)


    def evaluate_tensor(self, tensor: torch.Tensor) -> bool:
        if len(self.models) == 1:
            return self._evaluate_with_model(self.models[0], tensor)

        burp_votes = 0

        for model in self.models:
            if self._evaluate_with_model(model, tensor):
                burp_votes += 1

        return burp_votes


    def evaluate_array(self, array: np.ndarray) -> bool:
        tensor = prepare_array(array, self.sr)
        return self.evaluate_tensor(tensor)
