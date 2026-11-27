"""E3 (sensitivity), E4 (dependence-aware equivalence, substitute design), E5 (censoring simulation), E6 (window class).
All exploratory; inputs are the saved seed-2026 per-unit records."""
import numpy as np, pandas as pd
OUT = '<WORKDIR>/out'; MULTI = '<STUDY_REPO>/results/multi'
DS = ['PHM08', 'FD001', 'FD002', 'FD003', 'FD004', 'NCMAPSS6']; HS = [10, 15, 20, 30, 40]; LO, HI = -13, 2
WN = {ds: pd.read_csv(f'{MULTI}/{ds}_warnings.csv') for ds in DS}

def cost(g, l, rho):
    planned = g.pred.notna().values & (g.y.values > l)
    return np.where(planned, 1.0, rho), np.where(planned, g.cycle.values + l, g.life.values)

# ---------------- E3
RHO3 = [2, 3, 5, 7, 10, 15, 20, 30, 50, 75, 100, 150, 200]; LEAD3 = [0, 5, 10, 15, 20, 25, 30, 'U0-15', 'U0-30']
def ecost(g, l, rho):
    if isinstance(l, str):
        ls = range(0, 16) if l == 'U0-15' else range(0, 31); cs = [cost(g, x, rho) for x in ls]
        return np.mean([c for c, _ in cs], 0), np.mean([u for _, u in cs], 0)
    return cost(g, l, rho)
rows = []
for ds, W in WN.items():
    for fold in sorted(W.fold.unique()):
        trn = W[(W.fold == fold) & (W.role == 'train_oof')]; tst = W[(W.fold == fold) & (W.role == 'test')]
        Ltr = trn[trn.H == 20].life.values; Lte = tst[tst.H == 20].life.values
        for l in LEAD3:
            for rho in RHO3:
                bH = min(HS, key=lambda H: (lambda c, u: c.sum() / u.sum())(*ecost(trn[trn.H == H], l, rho)))
                c1, u1 = ecost(tst[tst.H == bH], l, rho); c2, u2 = ecost(tst[tst.H == 20], l, rho)
                Ts = np.arange(1, Ltr.max() + 1); cr = [np.where(Ltr > T, 1.0, rho).sum() / np.minimum(Ltr, T).sum() for T in Ts]; Ta = Ts[int(np.argmin(cr))]
                rows.append(dict(ds=ds, fold=fold, lead=str(l), rho=rho, rc=c1.sum(), ru=u1.sum(), hc=c2.sum(), hu=u2.sum(),
                                 ac=np.where(Lte > Ta, 1.0, rho).sum(), au=np.minimum(Lte, Ta).sum()))
E3 = pd.DataFrame(rows).groupby(['ds', 'lead', 'rho']).sum(numeric_only=True).reset_index()
E3['lr_costtuned_age'] = np.log((E3.rc / E3.ru) / (E3.ac / E3.au)); E3['lr_H20_age'] = np.log((E3.hc / E3.hu) / (E3.ac / E3.au))
E3.to_csv(f'{OUT}/E3_grid.csv', index=False)
be = []
for (ds, l), g in E3.groupby(['ds', 'lead']):
    g = g.sort_values('rho'); neg = g[g.lr_costtuned_age < 0]; neg20 = g[g.lr_H20_age < 0]
    be.append(dict(ds=ds, lead=l, breakeven_rho_costtuned=neg.rho.min() if len(neg) else np.nan,
                   costtuned_cheaper_all_rho_above=bool(len(neg) and (g[g.rho >= neg.rho.min()].lr_costtuned_age < 0).all()),
                   breakeven_rho_H20=neg20.rho.min() if len(neg20) else np.nan))
BE = pd.DataFrame(be); BE.to_csv(f'{OUT}/E3_breakeven.csv', index=False)
print(BE.pivot(index='ds', columns='lead', values='breakeven_rho_costtuned').to_string())
print(BE.pivot(index='ds', columns='lead', values='breakeven_rho_H20').to_string())

# ---------------- E4
CS = np.arange(0, 60.5, 0.5); urows = []
for ds, W in WN.items():
    for fold in sorted(W.fold.unique()):
        trn = W[(W.fold == fold) & (W.role == 'train_oof') & (W.H == 20)]; tst = W[(W.fold == fold) & (W.role == 'test') & (W.H == 20)]
        tw = trn[trn.pred.notna()]; sc = [((c - tw.y >= LO) & (c - tw.y <= HI)).sum() for c in CS]; c = CS[int(np.argmax(sc))]
        acc = tst.pred.notna() & (tst.pred - tst.y >= LO) & (tst.pred - tst.y <= HI)
        accc = tst.pred.notna() & (c - tst.y >= LO) & (c - tst.y <= HI)
        for a, b_, u in zip(acc, accc, tst.unit): urows.append(dict(ds=ds, fold=fold, unit=u, D=100 * (int(a) - int(b_)), c=c))
U4 = pd.DataFrame(urows); U4.to_csv(f'{OUT}/E4_units.csv', index=False)
z90 = 1.6449; n = len(U4); m = U4.D.mean(); se = U4.D.std(ddof=1) / np.sqrt(n)
rng = np.random.default_rng(9700); cl = [g.D.values for _, g in U4.groupby(['ds', 'fold'])]; bs = []
for _ in range(5000):
    pick = rng.integers(0, len(cl), len(cl)); v = np.concatenate([cl[i] for i in pick]); bs.append(v.mean())
d = U4.groupby('ds').D.agg(['mean', 'var', 'count']); vi = d['var'] / d['count']; wi = 1 / vi
mu_fe = (wi * d['mean']).sum() / wi.sum(); Q = (wi * (d['mean'] - mu_fe) ** 2).sum(); k = len(d)
tau2 = max(0.0, (Q - (k - 1)) / (wi.sum() - (wi ** 2).sum() / wi.sum())); ws = 1 / (vi + tau2); mu = (ws * d['mean']).sum() / ws.sum(); se_re = np.sqrt(1 / ws.sum())
E4 = pd.DataFrame([dict(method='unit-independent', est=m, lo90=m - z90 * se, hi90=m + z90 * se),
                   dict(method='cluster bootstrap (data set x fold, 30 clusters)', est=m, lo90=np.quantile(bs, .05), hi90=np.quantile(bs, .95)),
                   dict(method='random effects over data sets (DerSimonian-Laird)', est=mu, lo90=mu - z90 * se_re, hi90=mu + z90 * se_re, tau2=tau2)])
E4['equivalent_pm5'] = (E4.lo90 > -5) & (E4.hi90 < 5); E4.to_csv(f'{OUT}/E4_summary.csv', index=False)
d.assign(lo90=d['mean'] - z90 * np.sqrt(vi), hi90=d['mean'] + z90 * np.sqrt(vi)).to_csv(f'{OUT}/E4_per_ds.csv')
print(E4.round(3).to_string()); print(d.round(2).to_string())

# ---------------- E5
def km_censor_surv(T, delta):
    """Kaplan-Meier survival of the censoring time (censoring = event), returns function G(t-)"""
    ev = np.sort(np.unique(T[delta == 0])); S = 1.0; steps = []
    for t in ev:
        at = (T >= t).sum(); d_ = ((T == t) & (delta == 0)).sum(); S *= 1 - d_ / at; steps.append((t, S))
    st = np.array([t for t, _ in steps]); sv = np.array([s for _, s in steps])
    return lambda x: np.array([sv[st < xi][-1] if (st < xi).any() else 1.0 for xi in np.atleast_1d(x)])
rng = np.random.default_rng(9600); r5 = []
for ds, W in WN.items():
    g = W[(W.role == 'test') & (W.H == 20)]; L = g.life.values.astype(float); al = g.cycle.values
    acc = (g.pred.notna() & (g.pred - g.y >= LO) & (g.pred - g.y <= HI)).values.astype(float); truth = acc.mean()
    lo_, hi_ = 0.5 * np.median(L), 1.6 * L.max()
    for scen in ('independent', 'informative'):
        for rep in range(1000):
            C = rng.uniform(lo_, hi_, len(L))
            if scen == 'informative':
                pm = (~np.isnan(al)) & (rng.random(len(L)) < 0.5); C = np.where(pm, np.minimum(C, np.nan_to_num(al) + 15), C)
            T = np.minimum(L, C); delta = (L <= C).astype(int); G = km_censor_surv(T, delta)(T)
            terms = delta * acc / np.clip(G, 1e-6, None); est = terms.mean(); se = terms.std(ddof=1) / np.sqrt(len(L))
            naive = acc[delta == 1].mean() if delta.sum() else np.nan
            r5.append(dict(ds=ds, scen=scen, rep=rep, truth=truth, ipcw=est, cover=abs(est - truth) <= 1.96 * se, naive=naive, cens=1 - delta.mean()))
R5 = pd.DataFrame(r5); R5.to_csv(f'{OUT}/E5_reps.csv', index=False)
E5 = R5.groupby(['ds', 'scen']).apply(lambda x: pd.Series(dict(truth=100 * x.truth.iloc[0], censored=100 * x.cens.mean(),
        bias_ipcw=100 * (x.ipcw - x.truth).mean(), rmse_ipcw=100 * np.sqrt(((x.ipcw - x.truth) ** 2).mean()), coverage=100 * x.cover.mean(),
        bias_naive=100 * (x.naive - x.truth).mean()))).reset_index()
E5.to_csv(f'{OUT}/E5_summary.csv', index=False); print(E5.round(2).to_string())

# ---------------- E6
WINS = {'[-13,+2]': (-13, 2), '[-5,+5]': (-5, 5), '[-10,+10]': (-10, 10), '[-20,+20]': (-20, 20), '[-20,+5]': (-20, 5), '[-30,+10]': (-30, 10), '[-2,+13]': (-2, 13)}
r6 = []
for ds, W in WN.items():
    g = W[(W.role == 'test') & (W.H == 20)]; e = g.pred - g.y; warned = g.pred.notna()
    for name, (lo, hi) in list(WINS.items()) + [('lead-aware [-13,+2], R>l', (-13, 2))]:
        acc = warned & (e >= lo) & (e <= hi)
        row = dict(ds=ds, window=name, S=100 * acc.mean())
        for l in (0, 5, 10, 15):
            a = acc & (g.y > l) if name.startswith('lead-aware') else acc
            row[f'E{l}'] = 100 * (a & (g.y <= l)).mean(); row[f'S{l}' if name.startswith('lead-aware') else f'x{l}'] = 100 * a.mean()
        for l in (0, 5, 10, 15): row[f'F{l}'] = 100 * ((~warned) | (g.y <= l)).mean()
        r6.append(row)
E6 = pd.DataFrame(r6); E6.to_csv(f'{OUT}/E6_windows.csv', index=False)
WR = pd.concat([pd.read_csv(f'{MULTI}/{ds}_windows.csv').assign(ds=ds) for ds in DS]); WR = WR[WR.H == 20]
E6b = WR.groupby(['ds', 'lo', 'hi']).apply(lambda x: pd.Series({f'F{l}': 100 * ((x.pred.isna()) | (x.y <= l)).mean() for l in (0, 5, 10, 15)} | {'S': 100 * (x.cls == 'IN').mean()})).reset_index()
E6b.to_csv(f'{OUT}/E6_retuned.csv', index=False)
print(E6[['ds', 'window', 'S', 'E0', 'E5', 'E10', 'E15', 'F15']].round(1).to_string()); print(E6b.round(1).to_string())
