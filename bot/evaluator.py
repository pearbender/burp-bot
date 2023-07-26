from torch.utils.data import DataLoader, Dataset, random_split
import torch
import torchaudio
from torchaudio import transforms
import os
import random
import librosa
import soundfile as sf
import torch.nn.functional as F
from torch.nn import init
from torch import nn
import numpy as np
import shutil
import argparse
import math

from model import *


def prepare_file(audio_file: str) -> torch.Tensor:
    sig, sr = torchaudio.load(audio_file)
    spec = transforms.MelSpectrogram(
        sr, n_fft=1024, hop_length=None, n_mels=64)(sig)
    spec = transforms.AmplitudeToDB(top_db=80)(spec)

    return spec.unsqueeze(0)


class BurpEvaluatorModel:
    def __init__(self, model_filename: str) -> None:
        self.model = AudioClassifier()
        self.model.load_state_dict(torch.load(model_filename))
        self.model.eval()


    def _get_prediction(self, inputs: torch.Tensor):
        output = self.model.forward(inputs)
        conf, classes = torch.max(output, 1)
        return conf.item(), classes.item()


    def is_burp_in_tensor(self, tensor: torch.Tensor) -> bool:
        _, burp = self._get_prediction(tensor)
        return burp == 0


class MultiModelEvaluator:
    required_vote_fraction = 0.5
    evaluators = []


    def __init__(self, models) -> None:
        for model in models:
            self.evaluators += [BurpEvaluatorModel(model)]
    

    def is_burp_in_file(self, audio_file: str) -> bool:
        tensor = prepare_file(audio_file)
        burp_votes = 0
        
        for evaluator in self.evaluators:
            if evaluator.is_burp_in_tensor(tensor):
                burp_votes += 1
        
        return (burp_votes / len(self.evaluators)) >= self.required_vote_fraction
