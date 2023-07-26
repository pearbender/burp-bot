#!/bin/bash

if [ ! -d vods/audio ]; then
  mkdir -p vods/audio
fi

for vod in vods/*.mkv; do
  if [ ! -f "$vod" ]; then
    continue
  fi

  audio="vods/audio/$(basename $vod | sed -re 's/^(.*?)\.mkv$/vod_\1.flac/')"
  ffmpeg -i $vod -q:a 0 -map a $audio
done