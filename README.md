# Certifying the in-service reliability of end-of-life alarms for preventive replacement

Code, protocols and per-unit results for the article "Certifying the in-service reliability of end-of-life alarms for
preventive replacement", submitted to *Reliability Engineering & System Safety*.

## Layout
| Folder | Contents |
|---|---|
| `study/code` | The study pipeline: reference recipe (`eol_window2.py`, `multi_eval.py`, `safe_common.py`, `feats.py`), alarm families, battery and bearing pipelines, audits, figure scripts and `reproduce_*.sh` drivers. |
| `study/results` | Saved per-unit records (outer split seed 2026) and audit outputs used in the article (`multi/`, `audit/`, `window/`, `safe/`, `kam/`, `rev/`, `rep30/`). |
| `new_analyses` | Frozen-policy certification (E1), probabilistic-RUL baseline (E2), sensitivity (E3), dependence-aware equivalence (E4), censoring simulation (E5) and window class (E6): `protocol/` (protocol, SHA-256 hash list, frozen code, environment lock), `code/`, `results/`. |
| `field_validation` | Fleet-data application to SCANIA Component X (article Section 4.3, Supplementary Section S22): protocol v1 (hashed before the data were obtained), protocol v2 (written after the v1 results, hashed before it was run), deviation log, code and per-truck results (`RESULTS_SCANIA.md`; per-unit tables gzipped). The folder also holds scripts for Backblaze Drive Stats, which protocol v1 named as a second fleet dataset; **Backblaze was excluded before any file was downloaded, under the protocol's own admissibility rule, and no Backblaze result exists** (`DECISION_B1_backblaze.md`). |

## Data
The data are public and are not redistributed here. C-MAPSS (FD001–FD004), N-CMAPSS, PHM08 and FEMTO/PRONOSTIA are
listed in the NASA Prognostics Data Repository; the Severson et al. cells are available from the Toyota Research
Institute data platform (https://data.matr.io/1); the preprocessed XJTU, TJU, HUST and MIT battery files from
https://github.com/wang-fujin/PINN4SOH. Place them as described in `study/README_original.md`.

## Environment and commands
`pip install -r new_analyses/protocol/requirements.lock` (numpy 2.4.4, pandas 3.0.2, scikit-learn 1.8.0, scipy
1.17.1; other scikit-learn versions change the gradient-boosted predictions). Set `OMP_NUM_THREADS=1` when running fits
in parallel.

| Result | Command (from `new_analyses/code`) |
|---|---|
| Bit-identical reproduction of saved alarms | `python repro_check.py` |
| E1 fits and Table 6 | `bash run_fits.sh` then `python e1_analyze.py` |
| E2 fits and Table 7 | (fits in `run_fits.sh`) `python e2_analyze.py` |
| E3–E6 (Section 4.4, Supplementary S20) | `python e3_e6_analyze.py` (needs only `study/results/multi`) |
| Total-failure certificate (Proposition 3) and the α–λ acceptance rule (Section 4.3–4.4, Supplementary S22; derived/exploratory, added after review) | `python total_cert.py` (needs `study/results`, `new_analyses/results/e1_records_*` and `field_validation/results`; lifelines for the fleet part) |

The study's original tables and figures are produced by the `reproduce_*.sh` scripts in `study/code`.
Seeds: outer split 2026; E1 training 9100+s, inner 9200+s, calibration/test 9300, re-splits 9400+r; E2 bootstrap 9500;
E5 9600; E4 9700. The protocol in `new_analyses/protocol` was hashed before any new analysis
(hash of `SHA256SUMS`: 0b465f7c…); it was not registered publicly.

## Citation and licence
Please cite the article (details to follow on acceptance) and this repository.
Code: MIT licence (`LICENSE`). Data remain under their providers' terms.

## Provenance of the "frozen before running" statements
`FROZEN_AT.txt`, `protocol.sha256`, `protocol_v1_1.sha256` and `protocol_v2.sha256` record hashes and times written by the
analysts themselves; they are not third-party timestamps, and this repository's history was consolidated into a single
commit, so its date is not one either. The confirmatory study's protocol is separately registered on OSF Registries
(https://doi.org/10.17605/OSF.IO/Z78FM). The files in `new_analyses/protocol/frozen_code` are as hashed except
for redacted absolute paths in seven files (`frozen_code/PATHS_REDACTED.md` lists the lines and the new hashes).
