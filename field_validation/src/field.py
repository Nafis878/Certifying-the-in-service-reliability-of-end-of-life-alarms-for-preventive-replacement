"""Field validation of end-of-life alarm certification (protocol.md). One file, sub-commands:

  python src/field.py fit   <dataset> <seed>   train + freeze all policies on training units, log policy hashes
  python src/field.py score <dataset> <seed>   score frozen policies on calibration and test units (per-unit table)
  python src/field.py e1    <dataset>          field gap: S, F(l), E(l), U(l), S_N, cost vs optimal age
  python src/field.py e3    <dataset>          certification at scale (Algorithm 1), coverage, cluster coverage, total bound
  python src/field.py e4    <dataset>          transport: hold out whole model/spec groups
  python src/field.py e5    <dataset>          ranking: Kendall tau of S and S_N against realised cost; (F, U) frontier
Definitions follow the manuscript exactly: T = min Y - b, q_m = 1 - alpha^(1/m) (Algorithm 1, Theorem 1),
S_N(l, u) = share with l < R <= l + u (Eq. 6), C = (1 + (rho - 1) F) / (Lbar - U) (Eq. 11).
Datasets enter only through data/<ds>_panel.parquet (unit, t, features) and data/<ds>_units.parquet
(unit, entry, life, failed, cens_type, model, site)."""
import sys, json, hashlib, pickle, os
import numpy as np, pandas as pd
from scipy.stats import beta, kendalltau
from sklearn.ensemble import HistGradientBoostingRegressor
from lifelines import KaplanMeierFitter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFG = json.load(open(os.path.join(ROOT, 'config.json')))
ALPHA = CFG['alpha']

def load(ds):
    P = pd.read_parquet(f'{ROOT}/data/{ds}_panel.parquet'); U = pd.read_parquet(f'{ROOT}/data/{ds}_units.parquet')
    c = CFG['datasets'][ds]
    if c.get('top_models'):   # restrict to the most numerous model groups by UNIT COUNT (outcome-free)
        top = U.model.value_counts().index[:c['top_models']]; U = U[U.model.isin(top)]
    U = U.set_index('unit'); P = P[P.unit.isin(U.index)].sort_values(['unit', 't'])
    feats = [f for f in P.columns if f not in ('unit', 't')]
    return P, U, feats, c

def split_units(U, seed, c):
    """unit-level split, stratified by model group: train / calibration / test fractions from config"""
    rng = np.random.default_rng(seed); roles = {}
    for g, idx in U.groupby('model').groups.items():
        idx = np.array(sorted(idx)); rng.shuffle(idx); n = len(idx)
        a = int(round(c['f_train'] * n)); b = a + int(round(c['f_cal'] * n))
        roles.update({u: 'train' for u in idx[:a]}); roles.update({u: 'cal' for u in idx[a:b]}); roles.update({u: 'test' for u in idx[b:]})
    return pd.Series(roles, name='role')

def design(P, U, units, feats, cap):
    """training rows: failed units, every time step, RUL = life - t capped at cap; censored units only where the
    observed remaining time >= cap (their RUL is then known to exceed cap; label = cap)."""
    d = P[P.unit.isin(units)].copy(); life = d.unit.map(U.life); fail = d.unit.map(U.failed)
    rem = life - d.t; keep = (fail == 1) | (rem >= cap)
    d = d[keep]; y = np.minimum(rem[keep], cap)
    return d[feats + ['t']].to_numpy(float), y.to_numpy(float), d.unit.to_numpy()

def regressor(seed):
    return HistGradientBoostingRegressor(loss='absolute_error', max_iter=300, learning_rate=0.05, random_state=seed)

def predict_units(model, P, units, feats):
    d = P[P.unit.isin(units)]; p = model.predict(d[feats + ['t']].to_numpy(float))
    return {u: (g.t.to_numpy(), p[ix]) for (u, g), ix in zip(d.groupby('unit', sort=False), d.groupby('unit', sort=False).indices.values())}

def ewm(p, span):
    return pd.Series(p).ewm(span=span, adjust=False).mean().to_numpy() if span > 1 else p

def recipe_alarm(tp, H, span, k, shift):
    t, p = tp; q = ewm(p, span) + shift; hit = q <= H; run = 0
    for i, h in enumerate(hit):
        run = run + 1 if h else 0
        if run >= k: return t[i], float(q[i])
    return np.nan, np.nan

def window_ok(Y, R, lo, hi):
    return (~np.isnan(Y)) & (Y - R >= lo) & (Y - R <= hi)

# ------------------------------------------------------------------ policies
def fit_policies(ds, seed):
    P, U, feats, c = load(ds); roles = split_units(U, seed, c)
    tr = roles[roles == 'train'].index.to_numpy(); rng = np.random.default_rng(1000 + seed)
    folds = np.array_split(rng.permutation(tr), 5); H, lo, hi = c['H'], c['window'][0], c['window'][1]
    oof = {cap: {} for cap in c['caps']}
    for f in folds:
        fit_u = np.setdiff1d(tr, f)
        for cap in c['caps']:
            X, y, _ = design(P, U, fit_u, feats, cap); m = regressor(seed).fit(X, y); oof[cap].update(predict_units(m, P, f, feats))
    failed_tr = [u for u in tr if U.loc[u, 'failed'] == 1]
    best = None   # recipe selection: training OOF window score on failed training units (only they have R)
    for cap in c['caps']:
        for span in c['spans']:
            for k in c['ks']:
                for s in c['shifts']:
                    al = [recipe_alarm(oof[cap][u], H, span, k, s) for u in failed_tr]
                    t = np.array([a[0] for a in al]); Y = np.array([a[1] for a in al]); R = U.loc[failed_tr, 'life'].to_numpy() - t
                    sc = window_ok(Y, R, lo, hi).mean() - 1e-4 * abs(s)
                    if best is None or sc > best[0]: best = (sc, dict(cap=cap, span=span, k=k, shift=s, H=H))
    pol = {'recipe': best[1], 'naive_first_crossing': dict(cap=best[1]['cap'], span=1, k=1, shift=0.0, H=H)}
    models = {cap: regressor(seed).fit(*design(P, U, tr, feats, cap)[:2]) for cap in {best[1]['cap']}}
    # failure-probability policy (spirit of Mitici/de Pater): empirical OOF residual distribution in 10 bins
    cap = best[1]['cap']; allp, allr = [], []
    for u in failed_tr:
        t, p = oof[cap][u]; allp.append(p); allr.append(np.minimum(U.loc[u, 'life'] - t, cap) - p)
    allp, allr = np.concatenate(allp), np.concatenate(allr)
    edges = np.unique(np.quantile(allp, np.linspace(0, 1, 11)))
    pol['fail_prob'] = dict(cap=cap, edges=edges.tolist(), resid=[np.sort(allr[(np.clip(np.searchsorted(edges, allp, 'right') - 1, 0, len(edges) - 2)) == j]).tolist() for j in range(len(edges) - 1)],
                            pstar_grid=c['pstar'])
    # D1 (deviation log): p* chosen per (lead, rho) by the training out-of-fold renewal cost on failed training units
    sel = {}
    for lead in c['leads']:
        for rho in c['rhos']:
            best_p = None
            for ps in c['pstar']:
                cst = use = 0.0
                for u in failed_tr:
                    a, _ = fp_alarm(oof[cap][u], pol['fail_prob'], lead, ps); L = U.loc[u, 'life']
                    planned = (not np.isnan(a)) and (L - a > lead)
                    cst += 1.0 if planned else rho; use += (a + lead) if planned else L
                if best_p is None or cst / use < best_p[0]: best_p = (cst / use, ps)
            sel[f'{lead}_{rho}'] = best_p[1]
    pol['fail_prob']['pstar_sel'] = sel
    if c.get('smart_rule'):   # literature rule: any of the listed raw SMART counters becomes positive
        pol['smart_rule'] = dict(cols=c['smart_rule'])
    blob = pickle.dumps(dict(policies=pol, models=models, roles=roles, oof_failed=failed_tr))
    h = hashlib.sha256(blob).hexdigest(); os.makedirs(f'{ROOT}/results', exist_ok=True)
    open(f'{ROOT}/results/{ds}_s{seed}_frozen.pkl', 'wb').write(blob)
    with open(f'{ROOT}/results/policy_hashes.csv', 'a') as fh: fh.write(f'{ds},{seed},{h},{pd.Timestamp.utcnow().isoformat()}\n')
    roles.rename_axis('unit').reset_index().to_csv(f'{ROOT}/results/{ds}_s{seed}_roles.csv', index=False)
    print(ds, seed, 'frozen', h, best[1])

def fp_alarm(tp, pol, lead, pstar):
    t, p = tp; e = np.array(pol['edges']); bj = np.clip(np.searchsorted(e, p, 'right') - 1, 0, len(e) - 2); pr = np.zeros(len(p))
    for j in np.unique(bj):
        if '_resid_np' not in pol: pol['_resid_np'] = [np.asarray(x, float) for x in pol['resid']]   # D5: cache only
        r = pol['_resid_np'][j]; ix = bj == j; pr[ix] = np.searchsorted(r, lead + 1 - p[ix], 'right') / max(len(r), 1)
    i = np.flatnonzero(pr >= pstar)
    return (t[i[0]], float(p[i[0]])) if len(i) else (np.nan, np.nan)

def score(ds, seed):
    """Per-unit table for calibration and test units. Policies are loaded from the frozen file; its hash is checked."""
    P, U, feats, c = load(ds); blob = open(f'{ROOT}/results/{ds}_s{seed}_frozen.pkl', 'rb').read()
    h = hashlib.sha256(blob).hexdigest(); logged = pd.read_csv(f'{ROOT}/results/policy_hashes.csv', header=None)
    assert h in set(logged[2]), 'frozen policy hash not logged'
    F = pickle.loads(blob); pol, roles = F['policies'], F['roles']; units = roles[roles != 'train'].index
    pr = predict_units(F['models'][pol['recipe']['cap']], P, units, feats); rows = []
    for u in units:
        base = dict(ds=ds, seed=seed, unit=u, role=roles[u], life=U.loc[u, 'life'], failed=int(U.loc[u, 'failed']),
                    cens_type=U.loc[u, 'cens_type'], model=U.loc[u, 'model'], site=U.loc[u, 'site'])
        for name in ('recipe', 'naive_first_crossing'):
            q = pol[name]; a, Y = recipe_alarm(pr[u], q['H'], q['span'], q['k'], q['shift'])
            rows.append(base | dict(policy=name, alarm=a, Y=Y, lead_spec=-1))
        for lead in c['leads']:
            for ps in pol['fail_prob']['pstar_grid']:
                a, Y = fp_alarm(pr[u], pol['fail_prob'], lead, ps)
                rows.append(base | dict(policy=f'fail_prob_p{ps}_l{lead}', alarm=a, Y=Y, lead_spec=lead))
        if 'smart_rule' in pol:
            g = P[P.unit == u]; hit = (g[pol['smart_rule']['cols']].fillna(0) > 0).any(axis=1).to_numpy(); i = np.flatnonzero(hit)
            rows.append(base | dict(policy='smart_rule', alarm=g.t.to_numpy()[i[0]] if len(i) else np.nan, Y=np.nan, lead_spec=-1))
    T = pd.DataFrame(rows); T.to_csv(f'{ROOT}/results/{ds}_s{seed}_per_unit.csv', index=False); print(ds, seed, 'scored', len(T))

# ------------------------------------------------------------------ estimators
def km_censor(life, failed):
    k = KaplanMeierFitter().fit(life, event_observed=1 - failed)   # censoring as the event
    return lambda x: np.clip(k.survival_function_at_times(np.asarray(x, float) - 1e-9).to_numpy(), 1e-6, 1)

def outcome(df, lead):
    """In-service failure at lead l: no alarm before failure, or R <= l. Determined if failure observed, or the
    replacement time alarm + l falls before censoring. Returns (determined, fail, accepted-yet-failed, discarded, use)."""
    a = df.alarm.to_numpy(float); L = df.life.to_numpy(float); f = df.failed.to_numpy(int)
    warned = ~np.isnan(a) & (a < L); rep = np.where(warned, a + lead, np.inf)
    det_time = np.minimum(rep, np.where(f == 1, L, np.inf))
    determined = det_time <= L                     # censored units are determined only if replaced before censoring
    fail = np.where(determined, (f == 1) & (L <= rep), np.nan)
    R = np.where(warned & (f == 1), L - a, np.nan)
    disc = np.where(determined & (fail == 0), np.nan, 0.0)   # discarded life of planned replacements needs the failure time
    use = np.where(determined, np.minimum(det_time, L), np.nan)
    return determined, fail, R, disc, use, det_time

def ipcw_mean(x, det, det_time, G):
    w = np.where(det, 1 / G(np.where(det, det_time, 1)), 0.0); x = np.nan_to_num(x)
    return float((w * x).sum() / len(x))

def cp_upper(k, n, a=ALPHA):
    return 1.0 if n == 0 else float(beta.ppf(1 - a, k + 1, n - k)) if k < n else 1.0

def cluster_boot(df, fn, B, seed, by):
    rng = np.random.default_rng(seed); groups = [g for _, g in df.groupby(by)]; out = []
    for _ in range(B):
        pick = rng.integers(0, len(groups), len(groups)); out.append(fn(pd.concat([groups[i] for i in pick])))
    return np.nanquantile(out, [0.025, 0.975])

def optimal_age(train_life, train_failed, rho):
    k = KaplanMeierFitter().fit(train_life, event_observed=train_failed); ts = k.survival_function_.index.to_numpy()
    S = k.survival_function_.iloc[:, 0].to_numpy(); area = np.concatenate([[0], np.cumsum(np.diff(ts) * S[:-1])])
    cr = (S * 1 + (1 - S) * rho) / np.maximum(area, 1e-9); j = int(np.argmin(cr[1:]) + 1); return ts[j]

# ------------------------------------------------------------------ experiments
def e1(ds):
    c = CFG['datasets'][ds]; out = []
    for seed in c['seeds']:
        T = pd.read_csv(f'{ROOT}/results/{ds}_s{seed}_per_unit.csv'); T = T[T.role == 'test']
        G = km_censor(T.drop_duplicates('unit').life, T.drop_duplicates('unit').failed)
        sel = pickle.loads(open(f'{ROOT}/results/{ds}_s{seed}_frozen.pkl', 'rb').read())['policies']['fail_prob']['pstar_sel']
        for name, g in T.groupby('policy'):
            g = g.drop_duplicates('unit')
            if name.startswith('fail_prob'):
                lead0 = int(g.lead_spec.iloc[0])
                if name != f'fail_prob_p{sel[f"{lead0}_10"]}_l{lead0}': continue
            for lead in c['leads']:
                if g.lead_spec.iloc[0] not in (-1, lead): continue
                det, fail, R, _, _, dt = outcome(g, lead)
                Ff = lambda d: ipcw_mean(outcome(d, lead)[1], outcome(d, lead)[0], outcome(d, lead)[5], G)
                acc = window_ok(g.Y.to_numpy(float), R, *c['window'])
                E = (acc & (R <= lead)).astype(float); SN = ((R > lead) & (R <= lead + c['u'])).astype(float)
                k = int(np.nansum(fail[det])); n = int(det.sum())
                out.append(dict(ds=ds, seed=seed, policy=name, lead=lead, n=len(g), determined=n,
                                F_ipcw=Ff(g), F_ci=cluster_boot(g, Ff, c['B'], 9000 + seed, c['cluster']).tolist(),
                                F_cc=k / max(n, 1), F_cc_cp_upper=cp_upper(k, n),
                                S_failed_only=float(acc[g.failed == 1].mean()) if (g.failed == 1).any() else np.nan,
                                E_failed_only=float(E[g.failed == 1].mean()) if (g.failed == 1).any() else np.nan,
                                SN_failed_only=float(SN[g.failed == 1].mean()) if (g.failed == 1).any() else np.nan))
    pd.DataFrame(out).to_csv(f'{ROOT}/results/{ds}_E1.csv', index=False); print(pd.DataFrame(out).head(20).to_string())

def certify(cal_Y, b, k=1):
    y = np.sort(cal_Y[~np.isnan(cal_Y)]); m = len(y)
    return (y[k - 1] - b if m >= k else np.nan), m, 1 - ALPHA ** (1 / m) if m else np.nan

def e3(ds):
    c = CFG['datasets'][ds]; rows = []
    for seed in c['seeds']:
        T = pd.read_csv(f'{ROOT}/results/{ds}_s{seed}_per_unit.csv'); T = T[T.policy == 'recipe']
        cal, test = T[T.role == 'cal'], T[T.role == 'test']
        warned = lambda d: d[(~d.alarm.isna()) & (d.alarm < d.life)]
        Tlim, m, q = certify(warned(cal).Y.to_numpy(float), c['window'][1])
        tw = warned(test); exceed = float((tw.Y < Tlim + c['window'][1]).mean())   # outcome-free coverage event
        G = km_censor(test.life, test.failed); lead = int(np.floor(Tlim - 1e-9)) if Tlim > 0 else -1
        det, fail, R, _, _, dt = outcome(test, max(lead, 0))
        E = window_ok(test.Y.to_numpy(float), R, *c['window']) & (R <= lead)
        rows.append(dict(ds=ds, seed=seed, design='primary', m=m, meets_299=m >= 299, T=Tlim, q_m=q, certified_max_lead=lead,
                         test_exceed_share=exceed, viol=exceed > q, E_test_ipcw=ipcw_mean(E.astype(float), det, dt, G) if lead >= 0 else np.nan))
        # coverage over re-splits of the pooled calibration+test units (unit-level) and by cluster (dependence-aware)
        pool = pd.concat([cal, test]); rng = np.random.default_rng(7000 + seed)
        for design_, by in (('unit', 'unit'), ('cluster', c['cluster'])):
            keys = pool[by].unique(); viol = []
            for r in range(c['resplits']):
                kk = rng.permutation(keys); ca = pool[pool[by].isin(kk[:len(kk) // 2])]; te = pool[~pool[by].isin(kk[:len(kk) // 2])]
                Tl, mm, qq = certify(warned(ca).Y.to_numpy(float), c['window'][1])
                tw2 = warned(te); viol.append(float((tw2.Y < Tl + c['window'][1]).mean()) > qq)
            rows.append(dict(ds=ds, seed=seed, design=f'coverage_{design_}', viol_rate=float(np.mean(viol)), resplits=c['resplits']))
        # total-failure bound (protocol section D, Proposition F1): requires uncensored calibration outcomes
        calw = warned(cal); calR = (calw.life - calw.alarm)[calw.failed == 1].to_numpy(float)
        n_cal = len(cal); miss = int(((cal.alarm.isna()) | (cal.alarm >= cal.life))[cal.failed == 1].sum())
        if len(calR):
            TR = calR.min(); qR = 1 - ALPHA ** (1 / len(calR)); pm = cp_upper(miss, int((cal.failed == 1).sum()))
            rows.append(dict(ds=ds, seed=seed, design='total_bound_failed_only[NEEDS DERIVATION under censoring]',
                             T_R=TR, q_R=qR, miss_upper=pm, F_total_bound=pm + qR))
    pd.DataFrame(rows).to_csv(f'{ROOT}/results/{ds}_E3.csv', index=False); print(pd.DataFrame(rows).to_string())

def e4(ds):
    """transport: for each held-out model group, certify on calibration units of the other groups, test on the held-out
    group; eta_hat = observed exceedance share minus q_m (an estimate of the shift term of Theorem 1)."""
    c = CFG['datasets'][ds]; rows = []
    for seed in c['seeds']:
        T = pd.read_csv(f'{ROOT}/results/{ds}_s{seed}_per_unit.csv'); T = T[(T.policy == 'recipe') & (T.role != 'train')]
        w = T[(~T.alarm.isna()) & (T.alarm < T.life)]
        for g in w.model.unique():
            Tl, m, q = certify(w[(w.model != g) & (w.role == 'cal')].Y.to_numpy(float), c['window'][1])
            ex = float((w[w.model == g].Y < Tl + c['window'][1]).mean())
            rows.append(dict(ds=ds, seed=seed, held_out=g, m=m, T=Tl, q_m=q, exceed=ex, eta_hat=ex - q, n_held=int((w.model == g).sum())))
    pd.DataFrame(rows).to_csv(f'{ROOT}/results/{ds}_E4.csv', index=False); print(pd.DataFrame(rows).to_string())

def e5(ds):
    """does S_N (or S) rank realised cost? Kendall tau_b across policies within each (lead, rho) cell, test units,
    failed units only for S/S_N (they need R); cost from Eq. (11) with IPCW F and U."""
    c = CFG['datasets'][ds]; E1 = pd.read_csv(f'{ROOT}/results/{ds}_E1.csv'); out = []
    for (seed, lead), g in E1.groupby(['seed', 'lead']):
        for rho in c['rhos']:
            cost = (1 + (rho - 1) * g.F_ipcw)   # numerator of Eq. (11); denominator needs U, reported separately
            out.append(dict(seed=seed, lead=lead, rho=rho, tau_S=kendalltau(g.S_failed_only, -cost).statistic,
                            tau_SN=kendalltau(g.SN_failed_only, -cost).statistic))
    pd.DataFrame(out).to_csv(f'{ROOT}/results/{ds}_E5.csv', index=False); print(pd.DataFrame(out).groupby('lead').mean().to_string())

if __name__ == '__main__':
    cmd, ds = sys.argv[1], sys.argv[2]; seed = int(sys.argv[3]) if len(sys.argv) > 3 else None
    {'fit': lambda: fit_policies(ds, seed), 'score': lambda: score(ds, seed), 'e1': lambda: e1(ds), 'e3': lambda: e3(ds),
     'e4': lambda: e4(ds), 'e5': lambda: e5(ds)}[cmd]()
