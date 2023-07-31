import ffmpeg
import csv
import glob
import numpy as np
from io import BytesIO
import os
import random
import scipy.io.wavfile as wavfile
import soundfile as sf

vod_directory = "vods"
chat_logs_directory = "chat-logs"
burps_directory = "audio/burps"
maybe_burps_directory = "audio/maybe-burps"
not_burps_directory = "audio/not-burps"
vod_file_paths = glob.glob(os.path.join(vod_directory, '*.mkv'))


def seconds_to_time_string(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds_ = int(seconds % 60)
    milliseconds = int((seconds - int(seconds)) * 1000)
    time_string = f"{hours:02d}:{minutes:02d}:{seconds_:02d}.{milliseconds:03d}"
    return time_string


def clip_video(vod_file_path, start_time, end_time):
    audio_data, _ = (
        ffmpeg.input(vod_file_path, ss=seconds_to_time_string(
            start_time), to=seconds_to_time_string(end_time))
        .output("pipe:", format="wav", acodec="pcm_s16le", ac=2, ar="44100")
        .run(capture_stdout=True, capture_stderr=True)
    )
    buffer_io = BytesIO(audio_data)
    return wavfile.read(buffer_io)


def resize_to_1s(clip_samples, sample_rate):
    if len(clip_samples) > sample_rate:
        return clip_samples[:sample_rate]
    elif len(clip_samples) < sample_rate:
        return np.pad(clip_samples, ((0, sample_rate - len(clip_samples)), (0, 0)), mode='constant')
    return clip_samples


for vod_file_path in vod_file_paths:
    print(f"Clipping audio from {vod_file_path}...")
    name = os.path.splitext(os.path.basename(vod_file_path))[0]
    chat_log_file_path = f"{chat_logs_directory}/{name}.csv"

    try:
        with open(chat_log_file_path, newline='', encoding='utf-8') as csvfile:
            csv_data = csvfile.read()
            csv_rows = []
            lines = csv_data.splitlines()
            reader = csv.DictReader(lines)
            for csv_row in reader:
                csv_rows.append(csv_row)
    except FileNotFoundError:
        continue

    burp_end_times = []
    video_end_time = None
    for i, csv_row in enumerate(csv_rows):
        time = int(csv_row['time'])
        if i == len(csv_rows) - 1:
            video_end_time = time
        if csv_row['message'] != '!burrp':
            continue
        burp_end_times.append(time)

    burp_times = []
    for burp_end_time in burp_end_times:
        burp_start_time = max(0, burp_end_time - 15)
        burp_times.append((burp_start_time, burp_end_time))

    for (burp_start_time, burp_end_time) in burp_times:
        sample_rate, clip_samples = clip_video(
            vod_file_path, burp_start_time, burp_end_time)

        print(f"Found burp in [{burp_start_time}, {burp_end_time}].")

        clip_samples_mono = clip_samples.mean(axis=1)
        max_amplitude_sample = np.argmax(np.abs(clip_samples_mono))
        max_amplitude_time = burp_start_time + max_amplitude_sample / sample_rate
        precise_burp_start_time = max_amplitude_time - 0.5
        precise_burp_end_time = max_amplitude_time + 0.5

        sample_rate, clip_samples = clip_video(
            vod_file_path, precise_burp_start_time, precise_burp_end_time)
        clip_samples = resize_to_1s(clip_samples, sample_rate)

        print(
            f"Burp occurs at {max_amplitude_time}s.")

        clip_file_name = f"{name}_{int(round(max_amplitude_time))}.wav"
        clip_file_path = os.path.join(maybe_burps_directory, clip_file_name)
        burps_clip_file_path = os.path.join(burps_directory, clip_file_name)
        not_burps_clip_file_path = os.path.join(
            not_burps_directory, clip_file_name)

        if (os.path.exists(burps_clip_file_path) or os.path.exists(not_burps_clip_file_path)):
            print(f"Clip has already been labeled. Skipping.")
            continue
        print(f"Writing {clip_file_path}...")
        sf.write(clip_file_path, clip_samples, sample_rate)

    random.seed(42)

    for i in range(4 * len(burp_end_times)):
        while True:
            end_time = random.randint(1, video_end_time)
            start_time = end_time - 1
            for (burp_start_time, burp_end_time) in burp_times:
                if (start_time >= burp_start_time and start_time <= burp_end_time) or (end_time >= burp_start_time and end_time <= burp_end_time):
                    continue
            break

        sample_rate, clip_samples = clip_video(
            vod_file_path, start_time, end_time)
        clip_samples = resize_to_1s(clip_samples, sample_rate)
        clip_file_name = f"{name}_{start_time}.wav"
        clip_file_path = os.path.join(not_burps_directory, clip_file_name)

        if os.path.exists(clip_file_path):
            print(f"{clip_file_path} exists. Skipping.")
            continue

        print(f"Writing {clip_file_path}...")
        sf.write(clip_file_path, clip_samples, sample_rate)
