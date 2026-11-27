#!/usr/bin/env bash
# One command: bash reproduce.sh   (needs network to the hosts listed in README.md)
set -euo pipefail; cd "$(dirname "$0")"
python -m pip install -r requirements.lock
bash src/download_data.sh
python src/build_backblaze.py data; python src/build_scania.py data
sha256sum -c protocol.sha256
for ds in backblaze scania; do
  for s in 0 1 2; do python src/field.py fit $ds $s; python src/field.py score $ds $s; done
  python src/field.py e3 $ds; python src/field.py e1 $ds; python src/field.py e4 $ds; python src/field.py e5 $ds
done
