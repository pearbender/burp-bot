import ffmpeg
import glob
import math
from io import BytesIO
import moviepy.editor as mp
import numpy as np
import os
import scipy.io.wavfile as wavfile
import soundfile as sf
import sys
import torch
import torchaudio
from torchaudio import transforms

from model import *

model = AudioClassifier()
model.load_state_dict(torch.load("./model.pt"))
model.eval()

vod_directory = "audio/vods"
vod_file_paths = glob.glob(os.path.join(vod_directory, '*.pcm'))
burps_directory = "audio/burps"
maybe_burps_directory = "audio/maybe-burps"
not_burps_directory = "audio/not-burps"


def check_period(clip_samples, sample_rate):
    in_memory_file = BytesIO()
    wavfile.write(in_memory_file, sample_rate, clip_samples)

    sig, sr = torchaudio.load(in_memory_file)
    spec = transforms.MelSpectrogram(
        sr, n_fft=1024, hop_length=None, n_mels=64)(sig)
    spec = transforms.AmplitudeToDB(top_db=80)(spec)
    spec = spec.unsqueeze(0)
    output = model.forward(spec)
    conf, classes = torch.max(output, 1)
    is_burp = classes.item() == 0

    if is_burp and conf > 2.0:
        print(f"Found possible burp at {start_time}s.")
        destination_file_name = f"{name}_{int(start_time)}.wav"
        destination_file_path = os.path.join(
            maybe_burps_directory, destination_file_name)
        burps_clip_file_path = os.path.join(
            burps_directory, destination_file_name)
        not_burps_clip_file_path = os.path.join(
            not_burps_directory, destination_file_name)

        if os.path.exists(burps_clip_file_path) or os.path.exists(not_burps_clip_file_path):
            print(f"Clip has already been labeled. Skipping.\n")
            return

        if os.path.exists(destination_file_path):
            print(f"{destination_file_path} already exists. Skipping.")
            return

        print(f"Writing {destination_file_path}...\n")
        sf.write(destination_file_path, clip_samples, sample_rate)


for vod_file_path in vod_file_paths:
    print(f"Searching for burps in {vod_file_path}...")
    name = os.path.splitext(os.path.basename(vod_file_path))[0]

    with open(vod_file_path, "rb") as pcm_file:
        pcm_data = pcm_file.read()

    numpy_data = np.frombuffer(pcm_data, dtype=np.int16)
    samples = numpy_data.reshape(-1, 2)
    sample_rate = 44100
    duration = len(samples) // sample_rate

    for start_time in range(0, duration - 1):
        sample_start = start_time * sample_rate
        sample_end = sample_start + sample_rate
        start_time = sample_start / sample_rate
        end_time = sample_end / sample_rate
        clip_samples = samples[sample_start:sample_end, :]
        print(f"Checking [{start_time}, {end_time}]...", end="\r")
        sys.stdout.flush()
        check_period(clip_samples, sample_rate)

        sample_start = int(start_time * sample_rate + sample_rate / 2)
        sample_end = sample_start + sample_rate
        start_time = sample_start / sample_rate
        end_time = sample_end / sample_rate
        clip_samples = samples[sample_start:sample_end, :]
        print(f"Checking [{start_time}, {end_time}]...", end="\r")
        sys.stdout.flush()
        check_period(clip_samples, sample_rate)