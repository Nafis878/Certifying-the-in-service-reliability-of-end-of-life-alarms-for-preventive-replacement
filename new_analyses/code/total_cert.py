"""Exploratory/derived analyses added after review: (1) total-failure certificate (Proposition 3) on the E1 engine
records and on the SCANIA v2 records; (2) the alpha-lambda acceptance rule of Saxena et al. (2010) on the seed-2026
records. Not pre-specified."""
import os, glob, numpy as np, pandas as pd
from scipy.stats import beta
import pathlib
ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = str(ROOT / 'new_analyses' / 'results'); A1 = A2 = 0.025
MULTI = str(ROOT / 'study' / 'results')
cpu = lambda k, n, a: 1.0 if k >= n else float(beta.ppf(1 - a, k + 1, n - k))

# ---------- (1a) engines: complete data
rows = []
for f in sorted(glob.glob(f'{OUT}/e1_records_*_s*.csv')):
    R = pd.read_csv(f); ds, s = R.ds.iloc[0], int(R.seed.iloc[0]); pool = R[R.role == 'pool'].sort_values('unit').reset_index(drop=True)
    n = len(pool); h = n // 2
    def evalsplit(seed):
        p = np.random.default_rng(seed).permutation(n); c, t = pool.iloc[p[:h]], pool.iloc[p[h:]]
        miss = int(c.Y.isna().sum()); pbar = cpu(miss, len(c), A1); wR = c.R[c.Y.notna()].to_numpy(); m = len(wR)
        TR = wR.min(); q = 1 - A2 ** (1 / m); lmax = int(np.ceil(TR) - 1)
        Ft = float((t.Y.isna() | (t.R <= lmax)).mean()) if lmax >= 0 else np.nan
        S = float((t.Y.notna() & (t.Y - t.R >= -13) & (t.Y - t.R <= 2)).mean())
        return dict(ds=ds, seed=s, n_cal=len(c), missed=miss, p_bar=pbar, m=m, T_R=TR, q=q, bound=pbar + q, lmax=lmax,
                    F_test=Ft, viol=(Ft > pbar + q) if lmax >= 0 else np.nan, thm1_bound_dominated=1 - S)
    if s == 0: rows.append(dict(evalsplit(9300), design='primary'))
    cov = [evalsplit(9400 + r) for r in range(500)]
    rows.append(dict(ds=ds, seed=s, design='coverage', viol_rate=float(np.nanmean([c['viol'] for c in cov])),
                     bound_med=float(np.median([c['bound'] for c in cov])), lmax_med=float(np.median([c['lmax'] for c in cov]))))
T1 = pd.DataFrame(rows); T1.to_csv(f'{OUT}/TOTAL_engines.csv', index=False)
pd.set_option('display.width', 250); print(T1[T1.design == 'primary'].round(4).to_string())
print(T1[T1.design == 'coverage'].groupby('ds')[['viol_rate', 'bound_med', 'lmax_med']].mean().round(4).to_string())

# ---------- (1b) SCANIA v2: censored data, worst-case (valid for any censoring) and IPCW (independent censoring)
from lifelines import KaplanMeierFitter
FV = str(ROOT / 'field_validation' / 'results'); rows = []
for s in (0, 1, 2):
    T = pd.read_csv(f'{FV}/scania_v2_s{s}_per_unit.csv.gz'); c, t = T[T.role == 'cal'], T[T.role == 'test']
    warned = (~c.alarm.isna()) & (c.alarm < c.life)
    miss_wc = int((~warned).sum())                       # failed without alarm, or censored without alarm (worst case)
    Rstar = (c.life - c.alarm)[warned].to_numpy()        # notice if failed; time to censoring otherwise (<= true notice)
    pbar = cpu(miss_wc, len(c), A1); TR = Rstar.min(); q = 1 - A2 ** (1 / len(Rstar))
    # IPCW point estimate of the miss probability and of F at small leads on the test trucks
    k = KaplanMeierFitter().fit(t.life, event_observed=1 - t.failed); G = lambda x: np.clip(k.survival_function_at_times(np.asarray(x) - 1e-9).to_numpy(), 1e-6, 1)
    a = t.alarm.to_numpy(float); L = t.life.to_numpy(float); fl = t.failed.to_numpy(); w_ = ~np.isnan(a) & (a < L)
    missed_obs = (fl == 1) & ~w_; miss_ipcw = float((missed_obs / G(L)).sum() / len(t))
    rows.append(dict(seed=s, n_cal=len(c), worstcase_missed=miss_wc, worstcase_p_bar=pbar, m=len(Rstar), T_R_worstcase=TR, q=q,
                     worstcase_bound=pbar + q, test_miss_ipcw=miss_ipcw))
T2 = pd.DataFrame(rows); T2.to_csv(f'{OUT}/TOTAL_scania.csv', index=False); print(T2.round(4).to_string())

# ---------- (2) alpha-lambda acceptance at the alarm (Saxena et al. 2010), same alarms as the window analysis
rows = []
recs = []
for ds in ['PHM08', 'FD001', 'FD002', 'FD003', 'FD004', 'NCMAPSS6']:
    W = pd.read_csv(f'{MULTI}/multi/{ds}_warnings.csv'); g = W[(W.role == 'test') & (W.H == 20)]
    recs.append((ds, g.pred.to_numpy(float), g.y.to_numpy(float), [0, 5, 10, 15]))
B = pd.read_csv(f'{MULTI}/audit/cost_battery_nested_h60.csv'); B = B[(B.analysis == 'cv123') & (B.method == 'recipe')]
recs.append(('Severson', (B.pred * B.kappa).to_numpy(float), (B.y * B.kappa).to_numpy(float), [0, 15, 30, 45]))
for ds, Y, Rr, leads in recs:
    warned = ~np.isnan(Y)
    for al in (0.1, 0.2, 0.3):
        acc = warned & (np.abs(Y - Rr) <= al * Rr)
        row = dict(ds=ds, alpha=al, n=len(Y), S_alpha=100 * acc.mean())
        for i, l in enumerate(leads): row[f'E_l{i}'] = 100 * (acc & (Rr <= l)).mean(); row[f'F_l{i}'] = 100 * ((~warned) | (Rr <= l)).mean()
        rows.append(row)
T3 = pd.DataFrame(rows); T3.to_csv(f'{OUT}/ALPHA_LAMBDA.csv', index=False); print(T3.round(1).to_string())
