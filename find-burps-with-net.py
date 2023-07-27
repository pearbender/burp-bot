import torch
import torchaudio
from torchaudio import transforms
import os
import librosa
import soundfile as sf
import numpy as np
import shutil
import argparse
import math

from tqdm import tqdm
from tqdm import trange

from model_loader import BurpEvaluator


parser = argparse.ArgumentParser("burp-finder")
parser.add_argument("-t", "--template", help="Template file for length detection", type=str, required=True)
parser.add_argument("-o", "--output", help="Directory to store the outputs", type=str, default='./burp-found-temp')
parser.add_argument("-M", "--models", help="Model file[s] to use", type=str, default=['./model.pt'], nargs='+')
parser.add_argument("files", help="Audio files to parse", type=str, nargs='+')
args = parser.parse_args()


def get_template_size(template_file):
    audio_data, sr = librosa.load(template_file, sr=None, mono=False)
    return audio_data[0].size, sr


cut_size, cut_sr = get_template_size(args.template)

model = BurpEvaluator(args.models, sr=cut_sr)

print(f"Template size: {cut_size} samples, at rate {cut_sr}")


SLICE_OVERLAP = 4

MAX_FULL_SLICES_PER_CHUNK = 1000


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
        data, _ = librosa.load(audio_file, sr=sr, mono=False, offset=i / sr, duration=max_duration, dtype=np.float32)
        yield data


def cut_audio(audio_file, size, output):
    source, _ = os.path.splitext(os.path.basename(audio_file))

    current_slice = 0
    prev_slice_burp = False

    for chunk in load_file_chunks(audio_file, model.sr, size):
        for cut in get_cuts(chunk, size):
            # if current_slice % 500 == 0:
            #     tqdm.write(f"Processing slice {current_slice}...")
            
            if model.evaluate_array(cut):
                if prev_slice_burp:
                    tqdm.write(f"Burp found!! +++")
                else:
                    tqdm.write(f"Burp found!! {current_slice}")
                prev_slice_burp = True
                
                single_slice_file = f'{source}_slice_{current_slice}_from_{int(current_slice * (size // SLICE_OVERLAP) / model.sr)}.wav'
                sf.write(os.path.join(output, single_slice_file), cut.T, model.sr)
            else:
                prev_slice_burp = False

            current_slice += 1


if os.path.exists(args.output):
    shutil.rmtree(args.output)

os.makedirs(args.output)

print(f"\nPrepairing to parse {len(args.files)} files:")

for file in args.files:
    print(file)

for file in tqdm(args.files, unit='files', position=2, desc='Total files', dynamic_ncols=True, leave=False):
    tqdm.write(f"\n========================================\nParsing file {file}")
    cut_audio(file, cut_size, args.output)

print(f"Done")