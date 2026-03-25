#!/usr/bin/env bash
# Run on Ubuntu 22.04+ scraper VPS / RunPod CPU. Idempotent.
set -euo pipefail
apt-get update -y
apt-get install -y python3.11 python3.11-venv python3-pip git curl wget rsync
python3.11 -m venv /root/scraper_venv
# shellcheck source=/dev/null
source /root/scraper_venv/bin/activate
pip install --upgrade pip wheel
pip install -r "$(dirname "$0")/requirements-scraping.txt"
mkdir -p /workspace/eurobot_scraping/{esoteric,physics,math,astronomy,stem}/raw
mkdir -p /workspace/eurobot_scraping/output
echo "OK. Activate: source /root/scraper_venv/bin/activate"
echo "Copy scripts: rsync -av eurobot_baby/tools/scraping/*.py user@host:/workspace/eurobot_scraping/"
