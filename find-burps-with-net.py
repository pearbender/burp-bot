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

from tqdm import tqdm
from tqdm import trange

from model import *


model = AudioClassifier()
model.load_state_dict(torch.load("./model.pt"))
model.eval()


SLICE_OVERLAP = 4

MAX_FULL_SLICES_PER_CHUNK = 1000


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

    conf, classes = torch.max(output, 1)

    return conf.item(), classes.item()


def is_burp(audio_file):
    _, burp = get_prediction(audio_file)
    return burp == 0


def get_template_size(template_file):
    audio_data, sr = librosa.load(template_file, sr=None, mono=False)
    return audio_data[0].size, sr


def get_cuts(audio, size):
    for i in trange(0, audio[0].size - (size * (SLICE_OVERLAP - 1) // SLICE_OVERLAP), size // SLICE_OVERLAP, 
                    unit='slices', desc='Current chunk', dynamic_ncols=True, leave=False):
        yield np.array([audio[0][i : i + size], audio[1][i : i + size]])


def load_file_chunks(audio_file, sr, size):
    max_slices = MAX_FULL_SLICES_PER_CHUNK
    max_samples = size * max_slices
    max_duration = max_samples / sr

    duration = librosa.get_duration(path=audio_file)
    
    if duration < max_duration:
        data, _ = librosa.load(audio_file, sr=sr, mono=False)
        yield data
        return

    tqdm.write(f"File too big, loading in {math.ceil(duration / max_duration)} x {max_duration}s chunks")

    for i in trange(0, int(duration * sr), max_samples + 1, 
                    position=1, unit='chunks', desc='Current file', dynamic_ncols=True, leave=False):
        tqdm.write("Loading next chunk...")
        data, _ = librosa.load(audio_file, sr=sr, mono=False, offset=i / sr, duration=max_duration)
        yield data


def cut_audio(audio_file, size, sr, output):
    source, _ = os.path.splitext(os.path.basename(audio_file))

    current_slice = 0
    prev_slice_burp = False

    for chunk in load_file_chunks(audio_file, sr, size):
        for cut in get_cuts(chunk, size):
            # if current_slice % 500 == 0:
            #     tqdm.write(f"Processing slice {current_slice}...")
            
            single_slice_file = f'./burp-find-temp/{source}_slice_{current_slice}_from_{int(current_slice * (size // SLICE_OVERLAP) / sr)}.wav'
            sf.write(single_slice_file, cut.T, sr)
            
            if is_burp(single_slice_file):
                if prev_slice_burp:
                    tqdm.write(f"Burp found!! +++")
                else:
                    tqdm.write(f"Burp found!! {current_slice}")
                prev_slice_burp = True
                shutil.copy(single_slice_file, output)
            else:
                prev_slice_burp = False

            os.remove(single_slice_file)
            current_slice += 1


parser = argparse.ArgumentParser("burp-finder")
parser.add_argument("-t", "--template", help="Template file for length detection", type=str, required=True)
parser.add_argument("-o", "--output", help="Directory to store the outputs", type=str, default='./burp-found-temp')
parser.add_argument("files", help="Audio files to parse", type=str, nargs='+')
args = parser.parse_args()

cut_size, cut_sr = get_template_size(args.template)

print(f"Template size: {cut_size} samples, at rate {cut_sr}")

if os.path.exists('./burp-find-temp'):
    shutil.rmtree('./burp-find-temp')

if os.path.exists(args.output):
    shutil.rmtree(args.output)

os.makedirs('./burp-find-temp')
os.makedirs(args.output)

print(f"\nPrepairing to parse {len(args.files)} files:")

for file in args.files:
    print(file)

for file in tqdm(args.files, unit='files', position=2, desc='Total files', dynamic_ncols=True, leave=False):
    tqdm.write(f"\n========================================\nParsing file {file}")
    cut_audio(file, cut_size, cut_sr, args.output)

print(f"Done")