#!/bin/bash
VIDEO_QUALITY=160p

if [ ! -d chats/parsed ]; then
  mkdir -p chats/parsed
fi

if [ ! -d burps/parsed ]; then
  mkdir -p burps/parsed
fi

if [ ! -d burps-audio ]; then
  mkdir -p burps-audio
fi

if [ ! -d not-burps/parsed ]; then
  mkdir -p not-burps/parsed
fi

if [ ! -d not-burps-audio ]; then
  mkdir -p not-burps-audio
fi

for i in chats/*.csv; do
  id=`basename $i | sed -re 's/^[^0-9]*([0-9]*?)\.csv$/\1/'` # Get vod id from twitch chat csv filename
  vod="$id.mkv"
  echo $vod
  
  if [ ! -f "vods/$vod" ]; then
    echo -e "\n\nDownloading vod for $id..."
    twitch-dl download -q $VIDEO_QUALITY --auth-token $TWITCH_TOKEN "$id" --output "{id}.{format}" -w 32
    mv "$vod" vods/
  fi

  echo -e "\n\nClipping burps for $id using vods/$vod video and $i chat files..."

  python3 clip-burps.py "$i" "vods/$vod"
  
  echo -e "\n\nConverting burp clips to wav audio..."

  for burp in burps/*.mp4; do
    if [ ! -f "$burp" ]; then
      continue
    fi

    output=`basename $burp | sed -re 's/^(.*?)\.mp4$/\1.wav/'` # burps/smth.mp4 to smth.wav
   
    echo -e "\nConverting $burp to burps-audio/$output..."

    ffmpeg -i "$burp" -q:a 0 -map a "burps-audio/$output" -stats -loglevel warning -y
    mv "$burp" burps/parsed/
  done

  echo -e "\n\nConverting non burp clips to wav audio..."

  for burp in not-burps/*.mp4; do
    if [ ! -f "$burp" ]; then
      continue
    fi

    output=`basename $burp | sed -re 's/^(.*?)\.mp4$/\1.wav/'` # not-burps/smth.mp4 to smth.wav
   
    echo -e "\nConverting $burp to not-burps-audio/$output..."

    ffmpeg -i "$burp" -q:a 0 -map a "not-burps-audio/$output" -stats -loglevel warning -y
    mv "$burp" not-burps/parsed/
  done

  mv $i chats/parsed/
done
