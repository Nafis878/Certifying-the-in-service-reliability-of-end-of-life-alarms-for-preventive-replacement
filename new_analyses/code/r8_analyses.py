"""Derived analyses for revision round 8 (not pre-specified; computed from saved per-unit records).

(1) Theorem 3: monotone (fixed-sequence) certificate of the total in-service failure probability F(l), on the E1 engine
    records, with an i.i.d.-bootstrap coverage check that treats each data set's pool as the population.
(2) Proposition 4: simulation check of the optimality bound for distribution-free certificates.
(3) Window and lead-to-warning-level dependence of the accepted-yet-failed share (seed-2026 test records, five H levels).
(4) Proposition 5: partial-identification bounds and tipping points for F(l) on the censored SCANIA calibration trucks.
"""
import glob, numpy as np, pandas as pd, pathlib
from scipy.stats import beta
ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / 'new_analyses' / 'results'; MULTI = ROOT / 'study' / 'results' / 'multi'; FV = ROOT / 'field_validation' / 'results'
ALPHA = 0.05; LEADS = np.arange(0, 41)
cpu = lambda k, n, a=ALPHA: 1.0 if k >= n else float(beta.ppf(1 - a, k + 1, n - k))
pd.set_option('display.width', 250)

def fcurve(Y, R):
    """failure count at each lead: missed (no alarm) or notice <= l"""
    miss = np.isnan(Y); return np.array([int((miss | (~miss & (R <= l))).sum()) for l in LEADS])

def cert(Y, R, q):
    """Theorem 3: largest lead l such that the Clopper-Pearson bound of the failure count at l is <= q (-1: none)."""
    k = fcurve(Y, R); n = len(Y); ok = np.array([cpu(int(x), n) <= q for x in k])
    if not ok[0]: return -1, k
    return int(LEADS[np.argmin(ok)] - 1) if not ok.all() else int(LEADS[-1]), k

# ---------------------------------------------------------------- (1) engines
rows, cov = [], []
prop3 = pd.read_csv(OUT / 'TOTAL_engines.csv'); prop3 = prop3[prop3.design == 'primary'].set_index('ds')
for f in sorted(glob.glob(str(OUT / 'e1_records_*_s*.csv'))):
    P = pd.read_csv(f); ds, s = P.ds.iloc[0], int(P.seed.iloc[0])
    pool = P[P.role == 'pool'].sort_values('unit').reset_index(drop=True); n = len(pool); h = n // 2
    Yp, Rp = pool.Y.to_numpy(float), pool.R.to_numpy(float)
    if s == 0:
        p = np.random.default_rng(9300).permutation(n); c, t = p[:h], p[h:]
        for q in (0.05, 0.10):
            lh, k = cert(Yp[c], Rp[c], q)
            Ft = fcurve(Yp[t], Rp[t]) / len(t)
            row = dict(ds=ds, q=q, n_cal=len(c), n_test=len(t), lhat=lh, F_test_at_lhat=(Ft[lh] if lh >= 0 else np.nan))
            lm = int(prop3.loc[ds, 'lmax'])
            row.update(prop3_lmax=lm, prop3_bound=float(prop3.loc[ds, 'bound']), thm3_bound_at_prop3_lmax=cpu(int(k[lm]), len(c)))
            rows.append(row)
    # coverage: i.i.d. calibration samples of size h from the pool's empirical law (population F = pool F)
    Fpop = fcurve(Yp, Rp) / n; rng = np.random.default_rng(9800 + s)
    for q in (0.05, 0.10):
        v, lhs = [], []
        for r in range(2000):
            b = rng.integers(0, n, h); lh, _ = cert(Yp[b], Rp[b], q); lhs.append(lh)
            v.append(lh >= 0 and Fpop[lh] > q)
        cov.append(dict(ds=ds, seed=s, q=q, viol_rate=float(np.mean(v)), share_certified=float(np.mean(np.array(lhs) >= 0)),
                        lhat_median=float(np.median(lhs)), F_pop_l15=float(Fpop[15])))
T1 = pd.DataFrame(rows); T1.to_csv(OUT / 'R8_thm3_engines.csv', index=False); print(T1.round(4).to_string())
C1 = pd.DataFrame(cov); C1.to_csv(OUT / 'R8_thm3_coverage.csv', index=False)
print(C1.groupby(['ds', 'q'])[['viol_rate', 'share_certified', 'lhat_median']].mean().round(4).to_string())

# ---------------------------------------------------------------- (2) Proposition 4: optimality bound
rng = np.random.default_rng(9900); rows = []
for m in (30, 77, 150, 299):
    for qq in (0.01, 0.05):
        gamma = min(1.0, ALPHA / (1 - qq) ** m)
        Y = rng.standard_normal((20000, m)); y1 = Y.min(1)
        from scipy.stats import norm
        G = norm.cdf(y1); certify = rng.random(20000) < gamma
        rows.append(dict(m=m, q=qq, bound_alpha_over_1mq_m=ALPHA / (1 - qq) ** m, gamma=gamma,
                         viol_min_rule=float((G > qq).mean()), exact_min_rule=(1 - qq) ** m,
                         viol_randomized=float((certify & (G > qq)).mean()), P_certify_above_support_min=float(certify.mean())))
T2 = pd.DataFrame(rows); T2.to_csv(OUT / 'R8_prop4_sim.csv', index=False); print(T2.round(4).to_string())

# ---------------------------------------------------------------- (3) window and lead/H dependence
rows = []
for ds in ['PHM08', 'FD001', 'FD002', 'FD003', 'FD004', 'NCMAPSS6']:
    W = pd.read_csv(MULTI / f'{ds}_warnings.csv'); W = W[W.role == 'test']
    for H, g in W.groupby('H'):
        Y, R = g.pred.to_numpy(float), g.y.to_numpy(float); warned = ~np.isnan(Y); n = len(Y)
        for l in (0, 5, 10, 15):
            for name, a, b in (('task [-13,+2]', -13, 2), ('PHM08-constants [-13,+10]', -13, 10), ('symmetric [-7.5,+7.5]', -7.5, 7.5)):
                e = Y - R; acc = warned & (e >= a) & (e <= b)
                rows.append(dict(ds=ds, H=int(H), lead=l, ratio=l / H, window=name, n=n, S=100 * acc.mean(),
                                 E=100 * (acc & (R <= l)).mean(), F=100 * ((~warned) | (R <= l)).mean(),
                                 cond_minY_gt_l_plus_b=bool(np.nanmin(Y[warned]) > l + b)))
T3 = pd.DataFrame(rows); T3.to_csv(OUT / 'R8_window_ratio.csv', index=False)
T3['ratio_bin'] = pd.cut(T3.ratio, [-0.01, 0.0, 0.26, 0.51, 0.76, 1.6], labels=['0', '(0,.25]', '(.25,.5]', '(.5,.75]', '>.75'])
print(T3.groupby(['window', 'ratio_bin'], observed=True).agg(E_min=('E', 'min'), E_med=('E', 'median'), E_max=('E', 'max'),
      F_med=('F', 'median'), S_med=('S', 'median'), cond_share=('cond_minY_gt_l_plus_b', 'mean'), cells=('E', 'size')).round(1).to_string())

# ---------------------------------------------------------------- (4) SCANIA partial identification
rows = []
for s in (0, 1, 2):
    T = pd.read_csv(FV / f'scania_v2_s{s}_per_unit.csv.gz'); c = T[T.role == 'cal']
    a, L, fl = c.alarm.to_numpy(float), c.life.to_numpy(float), c.failed.to_numpy(int); n = len(c)
    warned = ~np.isnan(a) & (a < L)
    for l in (0, 10, 20):
        fail_known = ((fl == 1) & ~warned) | (warned & (fl == 1) & (L - a <= l))
        safe_known = warned & (L - a > l)          # observed alive more than l after the alarm
        unknown = ~(fail_known | safe_known)
        k, u = int(fail_known.sum()), int(unknown.sum())
        lam = [x for x in np.linspace(0, 1, 1001) if cpu(int(np.ceil(k + x * u)), n) <= 0.10]
        rows.append(dict(seed=s, lead=l, n=n, known_fail=k, unknown=u, known_safe=int(safe_known.sum()),
                         F_lower=k / n, F_upper=(k + u) / n, CP_upper_worst=cpu(k + u, n),
                         tipping_lambda_for_10pct=(max(lam) if lam else np.nan)))
T4 = pd.DataFrame(rows); T4.to_csv(OUT / 'R8_scania_partial_id.csv', index=False); print(T4.round(4).to_string())

# ---------------------------------------------------------------- (5) Proposition 6: cross-fitted certificates on the
# outer-fold test records (each fold's units are independent of the policy fitted on the other folds)
rows = []
for ds in ['PHM08', 'FD001', 'FD002', 'FD003', 'FD004', 'NCMAPSS6']:
    W = pd.read_csv(MULTI / f'{ds}_warnings.csv'); W = W[(W.role == 'test') & (W.H == 20)]; K = W.fold.nunique()
    for k, g in W.groupby('fold'):
        Y, R = g.pred.to_numpy(float), g.y.to_numpy(float); w_ = ~np.isnan(Y); m = int(w_.sum())
        T = float(np.nanmin(Y) - 2); qk = 1 - (ALPHA / K) ** (1 / m)
        k_ = fcurve(Y, R); ok = [cpu(int(x), len(Y), ALPHA / K) for x in k_]
        rows.append(dict(ds=ds, fold=int(k), n=len(Y), m=m, T=T, q_thm1=qk, thm3_bound_l0=ok[0], thm3_bound_l5=ok[5], thm3_bound_l10=ok[10]))
T5 = pd.DataFrame(rows); T5.to_csv(OUT / 'R8_crossfit.csv', index=False)
print(T5.groupby('ds').agg(K=('fold', 'size'), n_min=('n', 'min'), m_min=('m', 'min'), T_min=('T', 'min'), q_max=('q_thm1', 'max'),
      b0=('thm3_bound_l0', 'max'), b5=('thm3_bound_l5', 'max'), b10=('thm3_bound_l10', 'max')).round(3).to_string())

# ---------------------------------------------------------------- (6) Theorem 1: i.i.d.-bootstrap coverage of the 95%
# statement, treating each pool as the population: violation if Pr_pool{Y < min Y_cal | A = 1} > q_m
rows = []
for f in sorted(glob.glob(str(OUT / 'e1_records_*_s*.csv'))):
    P = pd.read_csv(f); ds, s = P.ds.iloc[0], int(P.seed.iloc[0]); pool = P[P.role == 'pool']; n = len(pool); h = n // 2
    Y = pool.Y.to_numpy(float); Yw = np.sort(Y[~np.isnan(Y)]); rng = np.random.default_rng(9850 + s); v = []
    for r in range(4000):
        b = Y[rng.integers(0, n, h)]; yw = b[~np.isnan(b)]; m = len(yw)
        if m == 0: continue
        qm = 1 - ALPHA ** (1 / m); v.append(np.searchsorted(Yw, yw.min(), 'left') / len(Yw) > qm)
    rows.append(dict(ds=ds, seed=s, viol_rate=float(np.mean(v))))
T6 = pd.DataFrame(rows); T6.to_csv(OUT / 'R8_thm1_bootstrap.csv', index=False); print(T6.groupby('ds').viol_rate.agg(['mean', 'max']).round(4).to_string())
