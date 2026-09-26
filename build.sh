#!/bin/bash
# Full pipeline: assets -> narration -> word alignment -> score -> frames -> final mp4
set -e
cd "$(dirname "$0")"
./fetch_assets.sh
mkdir -p tts chunks
python3 tts2.py                      # narration clips -> tts/*.wav + tts/lines.json
python3 align.py                     # measured word onsets -> tts/words.json (drives captions/slams)
python3 audio2.py synth              # synthesize all stems -> stems2.npz
python3 audio2.py mix voice=2.7 vfx=1.5 coda=1.05   # mix + master -> score2.wav
for r in "0 380" "380 760" "760 1140" "1140 1520" "1520 1900" "1900 2280"; do
  set -- $r
  python3 video2.py render $1 $2 chunks/$(printf "chunk_%04d.mp4" $1)
done
ls chunks/chunk_*.mp4 | sort | sed "s#^chunks/#file '#; s/$/'/" > chunks/list.txt
ffmpeg -y -loglevel error -f concat -safe 0 -i chunks/list.txt -i score2.wav -map 0:v -map 1:a \
  -c:v copy -c:a aac -b:a 256k -movflags +faststart -shortest the_good_word_v2_manga.mp4
echo "done -> the_good_word_v2_manga.mp4"
