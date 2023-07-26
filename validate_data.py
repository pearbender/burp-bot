# Moves out all clips that don't match the sample length and sample size of the template

import os
import shutil
import argparse
import librosa


def get_size_and_sr(filename):
    audio_data, sr = librosa.load(filename, sr=None, mono=False)
    return audio_data[0].size, sr


parser = argparse.ArgumentParser("data-validator")
parser.add_argument("template", help="Template file for length and sample rate detection", type=str)
args = parser.parse_args()


SIZE, SR = get_size_and_sr(args.template)


burps_folder_path = "./burps-audio"
burps_files = [os.path.join(burps_folder_path, file) for file in os.listdir(
    burps_folder_path) if file.lower().endswith(".wav")]

burps_folder_path = "./not-burps-audio"
not_burps_files = [os.path.join(burps_folder_path, file) for file in os.listdir(
    burps_folder_path) if file.lower().endswith(".wav")]


if os.path.exists('./eval-temp'):
    shutil.rmtree('./eval-temp')

os.makedirs('./eval-temp/wrong-burps')
os.makedirs('./eval-temp/wrong-non-burps')


print('\nfinding wrong-burps...')
for file in burps_files:
    size, sr = get_size_and_sr(file)
    if size != SIZE or sr != SR:
        print(file)
        shutil.move(file, './eval-temp/wrong-burps')


print('\nfinding wrong-burps...')
for file in not_burps_files:
    size, sr = get_size_and_sr(file)
    if size != SIZE or sr != SR:
        print(file)
        shutil.move(file, './eval-temp/wrong-non-burps')