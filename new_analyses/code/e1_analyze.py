"""E1 stage 2 (pre-specified): calibration, certified leads, H-E1a, H-E1b coverage, E1c trade-off, E1d k-th order, E1e."""
import os
import numpy as np, pandas as pd, glob, json
from scipy.stats import binom
OUT = os.environ.get('OUT_DIR', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results')); ALPHA, B, LO, HI = 0.05, 2.0, -13, 2; LEADS = np.arange(0, 31)
DS = ['PHM08', 'FD001', 'FD002', 'FD003', 'FD004']

def q_k(m, k):
    if m < k: return np.nan
    lo, hi = 0.0, 1.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if binom.cdf(k - 1, m, mid) <= ALPHA: hi = mid
        else: lo = mid
    return hi

def pi0(m, nt, q, k, rng):
    """P(share of test points below the k-th smallest calibration point > q) under exchangeability (Monte Carlo 20000)."""
    if m < k or nt == 0: return np.nan
    lab = np.r_[np.zeros(m), np.ones(nt)]; Mx = rng.permuted(np.tile(lab, (20000, 1)), axis=1)
    pos = np.argmax(np.cumsum(Mx == 0, axis=1) >= k, axis=1); cnt = pos - (k - 1)
    return float((cnt / nt > q).mean())

def evaluate(cal, test, k=1, msub=None):
    w = cal[cal.Y.notna()].sort_values('order')
    if msub is not None: w = w.iloc[:msub]
    m = len(w)
    if m < k: return dict(m=m, T=np.nan, q=np.nan)
    T = np.sort(w.Y.values)[k - 1] - B; q = q_k(m, k)
    acc = test.Y.notna() & (test.Y - test.R >= LO) & (test.Y - test.R <= HI)
    tw = test[test.Y.notna()]; pihat = float((tw.Y < T + B).mean()) if len(tw) else np.nan
    cert = LEADS[LEADS < T]
    E = {int(l): float((acc & (test.R <= l)).mean()) for l in cert}
    F = {int(l): float((test.Y.isna() | (test.R <= l)).mean()) for l in LEADS}
    return dict(m=m, T=float(T), q=q, lmax=int(cert.max()) if len(cert) else -1, Emax=max(E.values()) if E else np.nan,
                a_ok=(all(v <= q for v in E.values()) if E else None), pihat=pihat, viol=pihat > q, S=float(acc.mean()),
                F_lmax=F[int(cert.max())] if len(cert) else np.nan, nt=len(test), ntw=len(tw))

rng = np.random.default_rng(9999); prim, cov, trade, kres, oofc = [], [], [], [], []
for ds in DS:
    for f in sorted(glob.glob(f'{OUT}/e1_records_{ds}_s*.csv')):
        R = pd.read_csv(f); s = int(R.seed.iloc[0]); pool = R[R.role == 'pool'].sort_values('unit').reset_index(drop=True)
        n = len(pool); h = n // 2
        def split(seed):
            p = np.random.default_rng(seed).permutation(n); c = pool.iloc[p[:h]].copy(); c['order'] = np.arange(h)
            return c, pool.iloc[p[h:]].copy()
        if s == 0:
            c, t = split(9300)
            pd.DataFrame({'unit': np.r_[c.unit, t.unit, R[R.role == 'train_oof'].unit], 'role': ['cal'] * len(c) + ['test'] * len(t) + ['train'] * int((R.role == 'train_oof').sum())}).to_csv(f'{OUT}/e1_roles_{ds}_s0.csv', index=False)
            for k in (1, 2, 3):
                r = evaluate(c, t, k); r.update(ds=ds, k=k); (prim if k == 1 else kres).append(r)
            for ms in (10, 20, 40, None):
                r = evaluate(c, t, 1, ms); r.update(ds=ds, msub=ms or 'all'); trade.append(r)
            tro = R[(R.role == 'train_oof') & R.Y.notna()]; To = tro.Y.min() - B
            acc = t.Y.notna() & (t.Y - t.R >= LO) & (t.Y - t.R <= HI); cert = LEADS[LEADS < To]
            oofc.append(dict(ds=ds, T_oof=float(To), lmax_oof=int(cert.max()) if len(cert) else -1,
                             E_test_max=max([float((acc & (t.R <= l)).mean()) for l in cert]) if len(cert) else np.nan,
                             accepted_failures_at_lmax=int((acc & (t.R <= cert.max())).sum()) if len(cert) else 0, nt=len(t)))
        for r_ in range(500):
            c, t = split(9400 + r_)
            for k in (1, 2, 3):
                e = evaluate(c, t, k); e.update(ds=ds, seed=s, rep=r_, k=k); cov.append(e)
P = pd.DataFrame(prim); K = pd.DataFrame(kres); C = pd.DataFrame(cov); TR = pd.DataFrame(trade); OO = pd.DataFrame(oofc)
summ = []
for (ds, k), g in C.groupby(['ds', 'k']):
    g0 = g.iloc[0]; p0 = np.nanmean([pi0(int(r.m), int(r.ntw), r.q, k, rng) for r in g.sample(10, random_state=1).itertuples()])
    summ.append(dict(ds=ds, k=k, splits=len(g), share_T_pos=float((g['T'] > 0).mean()), T_med=float(g['T'].median()),
                     T_q10=float(g['T'].quantile(.1)), T_q90=float(g['T'].quantile(.9)), lmax_med=float(g.lmax.median()),
                     viol_rate=float(g.viol.mean()), pi0_mean=p0, criterion=float(g.viol.mean()) <= max(ALPHA, p0) + 0.03,
                     a_fail_rate=float((g.a_ok == False).mean()), q_mean=float(g.q.mean()), m_mean=float(g.m.mean())))
S = pd.DataFrame(summ)
for name, df in (('primary', P), ('kth', K), ('coverage_summary', S), ('tradeoff', TR), ('oof_comparison', OO), ('coverage_all', C)):
    df.to_csv(f'{OUT}/E1_{name}.csv', index=False)
print(P.round(4).to_string()); print(K.round(4).to_string()); print(S.round(4).to_string()); print(TR.round(4).to_string()); print(OO.round(4).to_string())
