#!/bin/bash
for i in $(ls burps/*.mp4);
  do output="burps-audio/$(basename -s '.mp4' $i).wav"
  echo $output
  ffmpeg -i "$i" -q:a 0 -map a "$output"
done

for i in $(ls not-burps/*.mp4);
  do output="not-burps-audio/$(basename -s '.mp4' $i).wav"
  echo $output
  ffmpeg -i "$i" -q:a 0 -map a "$output"
done