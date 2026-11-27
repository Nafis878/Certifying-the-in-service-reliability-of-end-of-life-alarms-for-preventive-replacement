"""SYNTHETIC data for a code smoke test ONLY. Never report results from it."""
import numpy as np, pandas as pd, os
rng = np.random.default_rng(1); rows = []; units = []
for u in range(400):
    life = int(rng.integers(80, 300)); fail = int(rng.random() < 0.6); obs = life if fail else int(rng.integers(40, life))
    for t in range(1, obs + 1):
        rows.append((u, t, (t / life) + rng.normal(0, .05), rng.normal()))
    units.append((u, 0, obs, fail, 'failure' if fail else 'administrative', f'm{u % 4}', 'NA'))
d = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
pd.DataFrame(rows, columns=['unit', 't', 'x1', 'x2']).to_parquet(f'{d}/smoketest_panel.parquet')
pd.DataFrame(units, columns=['unit', 'entry', 'life', 'failed', 'cens_type', 'model', 'site']).to_parquet(f'{d}/smoketest_units.parquet')
