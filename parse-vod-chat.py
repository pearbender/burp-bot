import csv
import sys
import os
from moviepy.video.io.VideoFileClip import VideoFileClip

chat_file_path = sys.argv[1]
vod_file_path = sys.argv[2]
name = os.path.splitext(os.path.basename(chat_file_path))[0]

with open(chat_file_path, newline='', encoding='utf-8') as csvfile:
    csv_data = csvfile.read()
    data = []
    lines = csv_data.splitlines()
    reader = csv.DictReader(lines)
    for row in reader:
        data.append(row)
for d in data:
    if d['message'] != '!burrp':
        continue
    print(d)
    end_time = int(d['time'])
    clip = VideoFileClip(vod_file_path).subclip(
        max(0, end_time - 30), end_time)
    # Change the filename and extension as needed
    output_path = f"burps/{name}_{d['time']}.mp4"
    clip.write_videofile(output_path, codec="libx264", audio_codec="aac")
    clip.close()
