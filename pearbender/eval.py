from torch.utils.data import DataLoader, Dataset, random_split
import torch
import torchaudio
from torchaudio import transforms
import os
import random
import torch.nn.functional as F
from torch.nn import init
from torch import nn
import shutil

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

    #inputs_m, inputs_s = inputs.mean(), inputs.std()
    #inputs = (inputs - inputs_m) / inputs_s

    output = model.forward(inputs)
    normalized_probs = F.softmax(output, dim=1)
    max_prob, class_index = torch.max(normalized_probs, dim=1)
    confidence = max_prob.item()
    predicted_class = class_index.item()
    return confidence, predicted_class


def is_burp(audio_file):
    _, burp = get_prediction(audio_file)
    return burp == 0


burps_folder_path = "./audio/burps"
burps_files = [os.path.join(burps_folder_path, file) for file in os.listdir(
    burps_folder_path) if file.lower().endswith(".wav")]

burps_folder_path = "./audio/not-burps"
not_burps_files = [os.path.join(burps_folder_path, file) for file in os.listdir(
    burps_folder_path) if file.lower().endswith(".wav")]


if os.path.exists('./eval-temp'):
    shutil.rmtree('./eval-temp')

os.makedirs('./eval-temp/false-positives')
os.makedirs('./eval-temp/false-negatives')


print('finding false-negatives...')
for burp in burps_files:
    if not is_burp(burp):
        print(f"{burp} {get_prediction(burp)}")
        shutil.copy(burp, './eval-temp/false-negatives')


print('finding false-positives...')
for burp in not_burps_files:
    if is_burp(burp):
        print(f"{burp} {get_prediction(burp)}")
        print(burp)
        shutil.copy(burp, './eval-temp/false-positives')


print("burps")
for i in range(10):
    file = random.choice(burps_files)
    print(f"{file} {get_prediction(file)}")

print("\n\n\nnon burps")
for i in range(10):
    file = random.choice(not_burps_files)
    print(f"{file} {get_prediction(file)}")