#!/bin/bash
# Downloads the open-source pixel fonts (SIL OFL, via google/fonts) and the two Piper voices.
set -e
cd "$(dirname "$0")"
mkdir -p fonts voices
for p in pressstart2p/PressStart2P-Regular.ttf vt323/VT323-Regular.ttf silkscreen/Silkscreen-Regular.ttf \
         silkscreen/Silkscreen-Bold.ttf dotgothic16/DotGothic16-Regular.ttf; do
  [ -f "fonts/$(basename $p)" ] || curl -fsSL -o "fonts/$(basename $p)" "https://raw.githubusercontent.com/google/fonts/main/ofl/$p"
done
cd voices
for v in voice-en-us-lessac-medium voice-en-us-ryan-high; do
  [ -f "${v#voice-}.onnx" ] || { curl -fsSL -o $v.tar.gz "https://github.com/rhasspy/piper/releases/download/v0.0.2/$v.tar.gz" && tar xzf $v.tar.gz && rm $v.tar.gz; }
done
ls
