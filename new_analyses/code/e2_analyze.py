"""E2 stage 2 (pre-specified): probabilistic failure-probability policy vs recipe (level chosen by training cost) vs age."""
import os
import numpy as np, pandas as pd, pickle
OUT = os.environ.get('OUT_DIR', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results')); MULTI = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'study', 'results', 'multi')
DS = ['PHM08', 'FD001', 'FD002', 'FD003', 'FD004', 'NCMAPSS6']; LEADS = [0, 5, 10, 15]; RHOS = [5, 10, 20, 50]
PSTAR = [0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50]; HS = [10, 15, 20, 30, 40]

def unit_cost(alarm, R, life, l, rho):
    """alarm: alarm cycle (nan if none); R: remaining life at alarm. Returns cost, operated cycles, failed, discarded life"""
    planned = (~np.isnan(alarm)) & (R > l)
    return np.where(planned, 1.0, rho), np.where(planned, alarm + l, life), ~planned, np.where(planned, R - l, 0.0)

def pp_alarms(oof, preds, l, ps):
    allp = np.concatenate([p for p, y in oof.values()]); allr = np.concatenate([y - p for p, y in oof.values()])
    edges = np.unique(np.quantile(allp, np.linspace(0, 1, 11))); bins = np.clip(np.searchsorted(edges, allp, 'right') - 1, 0, len(edges) - 2)
    res = [np.sort(allr[bins == j]) for j in range(len(edges) - 1)]
    out = {}
    for u, (p, y) in preds.items():
        bj = np.clip(np.searchsorted(edges, p, 'right') - 1, 0, len(edges) - 2)
        prob = np.zeros(len(p))
        for j in np.unique(bj):
            ix = bj == j; prob[ix] = np.searchsorted(res[j], l + 1 - p[ix], 'right') / max(len(res[j]), 1)
        out[u] = {}
        for s in ps:
            i = np.flatnonzero(prob >= s)
            out[u][s] = (i[0] + 1.0, float(y[i[0]])) if len(i) else (np.nan, np.nan)
    return out

rows = []   # per unit, per (l, rho), per policy
for ds in DS:
    D = pickle.load(open(f'{OUT}/e2_preds_{ds}.pkl', 'rb')); life = D['life']
    WN = pd.read_csv(f'{MULTI}/{ds}_warnings.csv')
    for fold, F in D['res'].items():
        trn = WN[(WN.fold == fold) & (WN.role == 'train_oof')]; tst = WN[(WN.fold == fold) & (WN.role == 'test')]
        Ltr = np.array([life[u] for u in F['train_u']]); Lte = {u: life[u] for u in F['test_u']}
        for l in LEADS:
            A_tr = pp_alarms(F['oof'], F['oof'], l, PSTAR); A_te = pp_alarms(F['oof'], F['test'], l, PSTAR)
            for rho in RHOS:
                def rate(alarms, s):
                    c = u_ = 0.0
                    for u, a in alarms.items():
                        cc, uu, _, _ = unit_cost(np.array([a[s][0]]), np.array([a[s][1]]), life[u], l, rho); c += cc[0]; u_ += uu[0]
                    return c / u_
                ps = min(PSTAR, key=lambda s: rate(A_tr, s))
                def rrate(H):
                    g = trn[trn.H == H]; cc, uu, _, _ = unit_cost(g.cycle.values, g.y.values, g.life.values, l, rho); return cc.sum() / uu.sum()
                bH = min(HS, key=rrate)
                Ts = np.arange(1, Ltr.max() + 1); cr = [np.where(Ltr > T, 1.0, rho).sum() / np.minimum(Ltr, T).sum() for T in Ts]; Ta = Ts[int(np.argmin(cr))]
                g = tst[tst.H == bH].set_index('unit')
                for u in F['test_u']:
                    L = Lte[u]
                    for pol, (al, R) in (('PP', A_te[u][ps]), ('recipe', (g.loc[u, 'cycle'], g.loc[u, 'y']))):
                        cc, uu, ff, dd = unit_cost(np.array([al], float), np.array([R], float), L, l, rho)
                        rows.append(dict(ds=ds, fold=fold, unit=u, lead=l, rho=rho, policy=pol, cost=cc[0], use=uu[0], fail=bool(ff[0]), disc=dd[0], sel=ps if pol == 'PP' else bH))
                    rows.append(dict(ds=ds, fold=fold, unit=u, lead=l, rho=rho, policy='age', cost=1.0 if L > Ta else rho, use=min(L, Ta), fail=L <= Ta, disc=max(L - Ta, 0), sel=Ta))
                    rows.append(dict(ds=ds, fold=fold, unit=u, lead=l, rho=rho, policy='perfect', cost=1.0, use=L - 1, fail=False, disc=0.0, sel=0))
    print(ds, 'done', flush=True)
U = pd.DataFrame(rows); U.to_csv(f'{OUT}/E2_units.csv', index=False)
cell = U.groupby(['ds', 'lead', 'rho', 'policy']).agg(cost=('cost', 'sum'), use=('use', 'sum'), F=('fail', 'mean'), Udisc=('disc', 'mean')).reset_index()
cell['rate'] = cell.cost / cell.use; cell.to_csv(f'{OUT}/E2_cells.csv', index=False)
def lr(c):
    p = c.pivot_table(index=['ds', 'lead', 'rho'], columns='policy', values='rate')
    return pd.DataFrame({'PP_age': np.log(p.PP / p.age), 'recipe_age': np.log(p.recipe / p.age), 'recipe_PP': np.log(p.recipe / p.PP)})
L0 = lr(cell); per_ds = L0.groupby('ds').mean(); pooled = L0.mean()
# cluster bootstrap: units within data set
rng = np.random.default_rng(9500); key = ['ds', 'lead', 'rho', 'policy']
arr = {ds: g for ds, g in U.groupby('ds')}; boots = []
for b in range(2000):
    parts = []
    for ds, g in arr.items():
        us = g.unit.unique(); pick = rng.choice(us, len(us)); cnt = pd.Series(pick).value_counts()
        gg = g.merge(cnt.rename('w'), left_on='unit', right_index=True)
        parts.append(gg.assign(cost=gg.cost * gg.w, use=gg.use * gg.w))
    bb = pd.concat(parts).groupby(key)[['cost', 'use']].sum().reset_index(); bb['rate'] = bb.cost / bb.use
    boots.append(lr(bb).mean())
Bt = pd.DataFrame(boots)
res = pd.DataFrame({'estimate': pooled, 'lo95': Bt.quantile(.025), 'hi95': Bt.quantile(.975)})
res.loc['H-E2a met'] = [float(res.loc['PP_age', 'hi95'] < 0)] * 3; res.loc['H-E2b met'] = [float(res.loc['recipe_PP', 'hi95'] < 0.05)] * 3
res.to_csv(f'{OUT}/E2_pooled.csv'); per_ds.to_csv(f'{OUT}/E2_per_ds.csv')
FU = cell.groupby(['ds', 'policy'])[['F', 'Udisc']].mean().unstack(); FU.to_csv(f'{OUT}/E2_FU.csv')
print(res.round(4).to_string()); print(per_ds.round(3).to_string()); print(FU.round(3).to_string())
