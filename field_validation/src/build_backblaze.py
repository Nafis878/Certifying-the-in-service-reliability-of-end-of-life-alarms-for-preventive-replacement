"""Builds the Backblaze unit-day panel and unit table (pre-specified in protocol.md section B1).
Unit = serial_number. Cohort = drives first observed >= 2023-02-01 (new installs; avoids left truncation).
Failure = failure == 1 on the drive's last record. Censoring: still reporting on the last day of the window
(administrative) or stops reporting without failure (removal: migration/decommission/other; mechanism unknown)."""
import glob, sys, zipfile, io, numpy as np, pandas as pd
D = sys.argv[1] if len(sys.argv) > 1 else 'data'
SMART = ['smart_5_raw', 'smart_187_raw', 'smart_188_raw', 'smart_197_raw', 'smart_198_raw', 'smart_9_raw', 'smart_194_raw',
         'smart_1_raw', 'smart_7_raw', 'smart_189_raw', 'smart_193_raw', 'smart_199_raw']
frames = []
for z in sorted(glob.glob(f'{D}/backblaze/data_Q*_20*.zip')):
    with zipfile.ZipFile(z) as zf:
        for n in sorted(zf.namelist()):
            if n.endswith('.csv') and '__MACOSX' not in n:
                df = pd.read_csv(io.BytesIO(zf.read(n)), usecols=lambda c: c in ['date', 'serial_number', 'model', 'failure', 'datacenter'] + SMART)
                frames.append(df)
P = pd.concat(frames, ignore_index=True); P['date'] = pd.to_datetime(P.date)
P = P.sort_values(['serial_number', 'date']).drop_duplicates(['serial_number', 'date'])
first = P.groupby('serial_number').date.min(); last = P.groupby('serial_number').date.max(); end = P.date.max()
keep = first[first >= pd.Timestamp('2023-02-01')].index
P = P[P.serial_number.isin(keep)].copy()
U = pd.DataFrame({'entry': first[keep], 'exit': last[keep]})
U['failed'] = P.groupby('serial_number').failure.max().reindex(keep).fillna(0).astype(int)
U['model'] = P.groupby('serial_number').model.first().reindex(keep)
U['site'] = P.groupby('serial_number').datacenter.first().reindex(keep) if 'datacenter' in P else 'NA'
U['cens_type'] = np.where(U.failed == 1, 'failure', np.where(U.exit >= end, 'administrative', 'removed_other'))
U['life'] = (U.exit - U.entry).dt.days + 1
P['t'] = (P.date - P.serial_number.map(U.entry)).dt.days + 1
P = P.rename(columns={'serial_number': 'unit'}); U.index.name = 'unit'
P[['unit', 't'] + [c for c in SMART if c in P]].to_parquet(f'{D}/backblaze_panel.parquet')
U.reset_index().to_parquet(f'{D}/backblaze_units.parquet')
print(U.cens_type.value_counts().to_string(), '\nunits', len(U), 'models', U.model.nunique(), 'window end', end.date())
