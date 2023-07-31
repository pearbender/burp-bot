import glob
from moviepy.editor import VideoFileClip, concatenate_videoclips
import os
import sys
import subprocess
import tempfile

burps_directory = "audio/burps"
file_paths = glob.glob(os.path.join(burps_directory, '*.wav'))
auth_token = sys.argv[1]


def to_time_string(hours, minutes):
    return f"{hours:02d}:{minutes:02d}"


burp_times = {}
for file_path in file_paths:
    name = os.path.splitext(os.path.basename(file_path))[0]
    name_parts = name.split('_')
    vod_id = name_parts[0]
    burp_time = int(name_parts[1]) - 2
    if not vod_id in burp_times:
        burp_times[vod_id] = []
    burp_times[vod_id].append(burp_time)

vod_ids = sorted(burp_times.keys())
video_clips = []

for vod_id in vod_ids:
    if vod_id == "1842686152":
        break
    print(f"Merging clips from {vod_id}...")
    burp_times[vod_id].sort()

    for burp_time in burp_times[vod_id]:
        print(f"  {burp_time}")
        before_burp_time = burp_time - 1

        before_burp_time_hours = int(before_burp_time // 3600)
        before_burp_time_minutes = int((before_burp_time % 3600) // 60)
        before_burp_time_seconds = int(before_burp_time % 60)

        start_time = before_burp_time_seconds
        end_time = before_burp_time_seconds + 3
        start_time_string = to_time_string(before_burp_time_hours, before_burp_time_minutes)
        if before_burp_time_minutes == 59:
            end_time_string = to_time_string(before_burp_time_hours + 1, 0)
        else:
            end_time_string = to_time_string(before_burp_time_hours, before_burp_time_minutes + 1)

        clip_file_name = tempfile.NamedTemporaryFile().name
        clip_file_name += '.mp4'

        command = f'twitch-dl download {vod_id} -q source -f mp4 --auth-token {auth_token} -o "{clip_file_name}" -s {start_time_string} -e {end_time_string}'
        print(f"  {command}")
        try:
            subprocess.run(command, shell=True, check=True, capture_output=True)
            clip = VideoFileClip(clip_file_name)
            clip = clip.subclip(start_time, end_time)
            video_clips.append(clip)
            #clip.close()
            #os.remove(clip_file_name)
        except subprocess.CalledProcessError as e:
            print("An error occurred:", e)
            break

final_clip = concatenate_videoclips(video_clips)
final_clip.write_videofile("burp-comp.mp4", codec="libx264")