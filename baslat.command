#!/usr/bin/env bash
# macklip'i başlatır (macOS / Linux). İlk çalıştırmada gerekenleri kendisi kurar.
cd "$(dirname "$0")"
set -e
if ! command -v ffmpeg >/dev/null; then
  echo "ffmpeg bulunamadı. Kur:  macOS 'brew install ffmpeg'  ·  Ubuntu 'sudo apt install ffmpeg'"
  exit 1
fi
if [ ! -d .venv ]; then
  echo "İlk kurulum yapılıyor (birkaç dakika sürer)..."
  python3 -m venv .venv
  .venv/bin/pip install -q -U pip
  .venv/bin/pip install -q -e .
fi
exec .venv/bin/macklip-arayuz
