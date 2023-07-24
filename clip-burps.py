import csv
import sys
import os
import random
from moviepy.video.io.VideoFileClip import VideoFileClip

chat_file_path = sys.argv[1]
vod_file_path = sys.argv[2]
name = os.path.splitext(os.path.basename(chat_file_path))[0]

def clip_video(dir, end_time):
    clip = VideoFileClip(vod_file_path).subclip(
        max(0, end_time - 30), end_time)
    output_path = f"{dir}/{name}_{str(end_time)}.mp4"
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

for end_time in end_times:
    clip_video("burps", end_time)

random.seed(42)
for i in range(len(end_times)):
    while True:
        end_time = random.randint(30, video_end_time)
        for other_end_time in end_times:
            if abs(other_end_time - end_time) <= 30:
                continue
        break
    clip_video("not-burps", end_time)
