# Field application: SCANIA Component X

Only SCANIA Component X was analysed (article Section 4.3, Supplementary Section S22; see RESULTS_SCANIA.md).
**Backblaze Drive Stats was excluded before any of its files were downloaded**, under protocol v1's own admissibility
rule (its failure flag includes operator-alarm and non-mechanical removals; see `DECISION_B1_backblaze.md`). No
Backblaze result exists; its scripts remain only because protocol v1 named them and its hash list covers them.

## Original package notes (written before any data were obtained)

The analysis workspace reached only PyPI and GitHub; every data host returned no connection (HTTP code 000):
f001.backblazeb2.com, www.backblaze.com, snd.se / researchdata.se, zenodo.org, figshare.com, kaggle.com,
huggingface.co, data.nasa.gov. Per hard rule 2, no field experiment was run and no field number exists.

To run: allow f001.backblazeb2.com, www.backblaze.com and snd.se (plus doi.org), or download the files yourself into
data/, then `bash reproduce.sh`. Runtime estimate: Backblaze 12 quarters is tens of GB unzipped; building the panel
needs >= 32 GB RAM or chunked reading; policy fits about 1-3 h per seed on 8 cores. SCANIA: under 1 h.

Contents: protocol.md (+ protocol.sha256, hashed before any data were seen), config.json (all pre-specified settings),
data_manifest.csv (to complete after download), src/ (download, dataset builders, field.py with fit/score/e1/e3/e4/e5),
requirements.lock. The code was smoke-tested end to end on synthetic data only (src/make_smoketest.py, 'smoketest'
entry in config.json); no synthetic output is kept.
