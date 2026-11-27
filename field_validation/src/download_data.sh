#!/usr/bin/env bash
# Downloads the two pre-specified field datasets and writes checksums. Needs outbound HTTPS to:
#   f001.backblazeb2.com (Backblaze Drive Stats quarterly ZIPs), www.backblaze.com (landing page / licence),
#   snd.se and doi.org (SCANIA Component X, SND-ID 2024-34, DOI 10.5878/jvb5-d390).
set -euo pipefail
cd "$(dirname "$0")/../data"
QUARTERS="${QUARTERS:-Q1_2023 Q2_2023 Q3_2023 Q4_2023 Q1_2024 Q2_2024 Q3_2024 Q4_2024 Q1_2025 Q2_2025 Q3_2025 Q4_2025}"
mkdir -p backblaze scania
for q in $QUARTERS; do
  f="backblaze/data_${q}.zip"
  [ -s "$f" ] || curl -fL --retry 3 -o "$f" "https://f001.backblazeb2.com/file/Backblaze-Hard-Drive-Data/data_${q}.zip"
done
# SCANIA Component X: SND does not publish stable per-file URLs in its catalogue entry; download "all files" from
# https://snd.se/en/catalogue/dataset/2024-34 (version 2) and place the CSVs (train_operational_readouts.csv,
# train_tte.csv, train_specifications.csv, validation_*.csv, test_*.csv) in data/scania/ before running build_scania.py.
ls scania/*.csv >/dev/null 2>&1 || echo "SCANIA files missing: download manually into data/scania/ (see comment)."
date -u +%Y-%m-%dT%H:%M:%SZ > DOWNLOADED_AT.txt
find . -type f \( -name '*.zip' -o -name '*.csv' \) -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS.data
echo "checksums written to data/SHA256SUMS.data"
