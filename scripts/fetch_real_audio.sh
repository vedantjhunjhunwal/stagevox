#!/usr/bin/env bash
# Fetch the held-out real recordings (LibriSpeech excerpts + CC/PD music) from librosa's data repo.
set -euo pipefail
DEST=${1:-ext/librosa-data}
git clone --depth 1 --filter=blob:none --sparse https://github.com/librosa/data.git "$DEST"
git -C "$DEST" sparse-checkout set audio
echo "real audio in $DEST/audio"
