import ffmpeg
import csv
import sys
import os
from moviepy.video.io.VideoFileClip import VideoFileClip
import numpy as np
import scipy.io.wavfile as wavfile
from io import BytesIO
import random


chat_file_path = sys.argv[1]
vod_file_path = sys.argv[2]
name = os.path.splitext(os.path.basename(chat_file_path))[0]
clip_length = 4


def convert_seconds_to_hhmmss(seconds):
    hh = seconds // 3600
    mm = (seconds % 3600) // 60
    ss = seconds % 60
    return f"{hh:02d}:{mm:02d}:{ss:02d}"


def clip_burp(end_time_seconds):
    start_time_seconds = max(0, end_time_seconds - 15)
    start_time = convert_seconds_to_hhmmss(start_time_seconds)
    end_time = convert_seconds_to_hhmmss(end_time_seconds)
    audio_data, _ = (
        ffmpeg.input(vod_file_path, ss=start_time, to=end_time)
        .output("pipe:", format="wav", acodec="pcm_s16le", ac=1, ar="44100")
        .run(capture_stdout=True, capture_stderr=True)
    )

    buffer_io = BytesIO(audio_data)
    sample_rate, audio_samples = wavfile.read(buffer_io)

    # Check if the WAV file is stereo (2 channels) and convert to mono (average channels)
    if len(audio_samples.shape) > 1 and audio_samples.shape[1] == 2:
        audio_samples = np.mean(audio_samples, axis=1)

    time = np.linspace(start_time_seconds,
                       end_time_seconds, len(audio_samples))
    burp_sample = np.argmax(np.abs(audio_samples))
    burp_time_seconds = time[burp_sample]

    clip_length = 4.0
    before_time_seconds = random.uniform(0.5, clip_length - 0.5)
    before_burp_time_seconds = burp_time_seconds - before_time_seconds
    after_burp_time_seconds = burp_time_seconds + \
        (clip_length - before_time_seconds)
    return (before_burp_time_seconds, after_burp_time_seconds)


def clip_video(dir, start_time, end_time):
    clip = VideoFileClip(vod_file_path).subclip(start_time, end_time)
    output_path = f"{dir}/{name}_{str(int(start_time))}.mp4"
    print(f"Writing {output_path}")
    clip.write_videofile(output_path, codec="libx264", audio_codec="aac")
    clip.close()


with open(chat_file_path, newline='', encoding='utf-8') as csvfile:
    csv_data = csvfile.read()
    data = []
    lines = csv_data.splitlines()
    reader = csv.DictReader(lines)
    for row in reader:
        data.append(row)

end_times = []
video_end_time = None
for i, d in enumerate(data):
    end_time = int(d['time'])
    if i == len(data) - 1:
        video_end_time = end_time
    if d['message'] != '!burrp':
        continue
    end_times.append(end_time)

random.seed(42)

burps = []
for end_time in end_times:
    burps.append(clip_burp(end_time))

for burp in burps:
    (burp_start_time, burp_end_time) = burp
    clip_video("burps", burp_start_time, burp_end_time)

for i in range(4 * len(end_times)):
    while True:
        end_time = random.randint(clip_length, video_end_time)
        start_time = end_time - clip_length
        for burp in burps:
            (burp_start_time, burp_end_time) = burp
            if (start_time >= burp_start_time and start_time <= burp_end_time) or (end_time >= burp_start_time and end_time <= burp_end_time):
                continue
        break
    clip_video("not-burps", start_time, end_time)
