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

from model import *


model = AudioClassifier()
model.load_state_dict(torch.load("./model.pt"))
model.eval()


def prepare_file(audio_file):
    sig, sr = torchaudio.load(audio_file)
    spec = transforms.MelSpectrogram(
        sr, n_fft=1024, hop_length=None, n_mels=64)(sig)
    spec = transforms.AmplitudeToDB(top_db=80)(spec)

    return spec.unsqueeze(0)


def get_prediction(audio_file):
    inputs = prepare_file(audio_file)
    
    inputs_m, inputs_s = inputs.mean(), inputs.std()
    inputs = (inputs - inputs_m) / inputs_s
    
    output = model.forward(inputs)

    conf, classes = torch.max(output, 1)

    return conf.item(), classes.item()


def is_burp(audio_file):
    _, burp = get_prediction(audio_file)
    return burp == 0


def get_template_size(template_file):
    audio_data, _ = librosa.load(template_file, sr=None, mono=False)
    return audio_data[0].size


def get_cuts(audio, size):
    for i in range(0, audio[0].size - size // 2, size // 2):
        yield np.array([audio[0][i : i + size], audio[1][i : i + size]])


def cut_audio(audio_file, size):
    if os.path.exists('./burp-find-temp'):
        shutil.rmtree('./burp-find-temp')

    if os.path.exists('./burp-found-temp'):
        shutil.rmtree('./burp-found-temp')

    os.makedirs('./burp-find-temp')
    os.makedirs('./burp-found-temp')

    audio_data, sr = librosa.load(audio_file, sr=None, mono=False)
    
    current_slice = 0
    for cut in get_cuts(audio_data, size):
        print(f"Processing slice {current_slice}...")
        single_slice_file = f'./burp-find-temp/slice{current_slice}.wav'
        sf.write(single_slice_file, cut.T, sr)
        if is_burp(single_slice_file):
            print(f"Burp found!!")
            shutil.copy(single_slice_file, './burp-found-temp/')
        os.remove(single_slice_file)
        current_slice += 1


parser = argparse.ArgumentParser("burp-finder")
parser.add_argument("template", help="Template file for length detection", type=str)
parser.add_argument("file", help="Audio file to parse", type=str)
args = parser.parse_args()

cut_size = get_template_size(args.template)

print(f"Template size: {cut_size} samples")

cut_audio(args.file, cut_size)