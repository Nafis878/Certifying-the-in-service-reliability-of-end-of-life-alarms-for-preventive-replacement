"""Field protocol v2: hazard alarm (protocol_v2.md). Usage: python src/field_v2.py run scania <seed>; python src/field_v2.py report scania"""
import sys, os, pickle, hashlib, numpy as np, pandas as pd
from scipy.stats import binom
from sklearn.ensemble import HistGradientBoostingClassifier
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from field import ROOT, CFG, ALPHA, load, predict_units, outcome, km_censor, ipcw_mean, cp_upper, window_ok, certify
THETAS = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]; HZ = 20

def labelled_rows(P, U, units, feats):
    d = P[P.unit.isin(units)]; L = d.unit.map(U.life).to_numpy(); f = d.unit.map(U.failed).to_numpy(); t = d.t.to_numpy()
    y = np.where((f == 1) & (L - t <= HZ), 1, np.where(L - t > HZ, 0, -1)); keep = y >= 0
    return d[feats + ['t']].to_numpy(float)[keep], y[keep]

def clf(seed): return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=seed)

def proba_units(m, P, units, feats):
    d = P[P.unit.isin(units)]; p = m.predict_proba(d[feats + ['t']].to_numpy(float))[:, 1]
    return {u: (g.t.to_numpy(), p[ix]) for (u, g), ix in zip(d.groupby('unit', sort=False), d.groupby('unit', sort=False).indices.values())}

def first_alarm(tp, theta):
    t, p = tp; i = np.flatnonzero(p >= theta); return t[i[0]] if len(i) else np.nan

def ipcw_cost(df, lead, rho):
    G = km_censor(df.life, df.failed); det, fail, R, _, use, dt = outcome(df, lead)
    w = np.where(det, 1 / G(np.where(det, dt, 1)), 0.0); cost = np.where(det, np.where(np.nan_to_num(fail) == 1, rho, 1.0), 0.0)
    return float((w * cost).sum() / (w * np.nan_to_num(use)).sum())

def run(ds, seed):
    P, U, feats, c = load(ds); F1 = pickle.loads(open(f'{ROOT}/results/{ds}_s{seed}_frozen.pkl', 'rb').read())
    roles = F1['roles']; tr = roles[roles == 'train'].index.to_numpy(); folds = np.array_split(np.random.default_rng(1000 + seed).permutation(tr), 5)
    oof = {}
    for f in folds: oof.update(proba_units(clf(seed).fit(*labelled_rows(P, U, np.setdiff1d(tr, f), feats)), P, f, feats))
    base = U.loc[tr, ['life', 'failed']].reset_index()
    best = min(THETAS, key=lambda th: ipcw_cost(base.assign(alarm=[first_alarm(oof[u], th) for u in base.unit]), 7, 10))
    model = clf(seed).fit(*labelled_rows(P, U, tr, feats))
    blob = pickle.dumps(dict(model=model, theta=best)); h = hashlib.sha256(blob).hexdigest()
    open(f'{ROOT}/results/{ds}_v2_s{seed}_frozen.pkl', 'wb').write(blob)
    with open(f'{ROOT}/results/policy_hashes.csv', 'a') as fh: fh.write(f'{ds}_v2,{seed},{h},{pd.Timestamp.utcnow().isoformat()}\n')
    units = roles[roles != 'train'].index; pr = proba_units(model, P, units, feats)
    reg = F1['models'][F1['policies']['recipe']['cap']]; rp = predict_units(reg, P, units, feats); rows = []
    for u in units:
        a = first_alarm(pr[u], best); Y = np.nan
        if not np.isnan(a): Y = float(rp[u][1][np.flatnonzero(rp[u][0] == a)[0]])
        rows.append(dict(ds=ds, seed=seed, unit=u, role=roles[u], life=U.loc[u, 'life'], failed=int(U.loc[u, 'failed']),
                         model=U.loc[u, 'model'], policy='hazard_v2', alarm=a, Y=Y))
    pd.DataFrame(rows).to_csv(f'{ROOT}/results/{ds}_v2_s{seed}_per_unit.csv', index=False); print(ds, seed, 'v2 frozen', h, 'theta', best)

def q_k(m, k):
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2; lo, hi = (lo, mid) if binom.cdf(k - 1, m, mid) <= ALPHA else (mid, hi)
    return hi

def report(ds):
    c = CFG['datasets'][ds]; b = c['window'][1]; e3, e1 = [], []
    for seed in c['seeds']:
        T = pd.read_csv(f'{ROOT}/results/{ds}_v2_s{seed}_per_unit.csv'); cal, test = T[T.role == 'cal'], T[T.role == 'test']
        warned = lambda d: d[(~d.alarm.isna()) & (d.alarm < d.life)]
        y = np.sort(warned(cal).Y.dropna().to_numpy()); m = len(y); tw = warned(test)
        G = km_censor(test.life, test.failed)
        for k in (1, 2, 3):
            if m < k: e3.append(dict(seed=seed, k=k, m=m)); continue
            Tl = y[k - 1] - b; q = 1 - ALPHA ** (1 / m) if k == 1 else q_k(m, k); lead = int(np.floor(Tl - 1e-9)) if Tl > 0 else -1
            ex = float((tw.Y < y[k - 1]).mean()) if len(tw) else np.nan
            Et = np.nan
            if lead >= 0:
                det, fail, R, _, _, dt = outcome(test, lead); E = window_ok(test.Y.to_numpy(float), R, *c['window']) & (R <= lead)
                Et = ipcw_mean(E.astype(float), det, dt, G)
            rng = np.random.default_rng(7000 + seed); pool = pd.concat([cal, test]); keys = pool.unit.unique(); viol = []
            for _ in range(c['resplits']):
                kk = rng.permutation(keys); ca = warned(pool[pool.unit.isin(kk[:len(kk) // 2])]); te = warned(pool[~pool.unit.isin(kk[:len(kk) // 2])])
                yy = np.sort(ca.Y.dropna().to_numpy())
                if len(yy) < k: continue
                qq = 1 - ALPHA ** (1 / len(yy)) if k == 1 else q_k(len(yy), k); viol.append(float((te.Y < yy[k - 1]).mean()) > qq)
            e3.append(dict(seed=seed, k=k, m=m, meets_299=m >= 299, T=Tl, q=q, certified_max_lead=lead, test_exceed=ex,
                           E_test_ipcw_at_lmax=Et, viol_rate_unit=float(np.mean(viol)) if viol else np.nan))
        for lead in c['leads']:
            det, fail, R, _, use, dt = outcome(test, lead); acc = window_ok(test.Y.to_numpy(float), R, *c['window'])
            fu = test.failed.to_numpy() == 1
            e1.append(dict(seed=seed, lead=lead, alarmed=int(((~test.alarm.isna()) & (test.alarm < test.life)).sum()), n=len(test),
                           failed=int(fu.sum()), warned_before_repair=int((~np.isnan(R) & fu).sum()),
                           F_ipcw=ipcw_mean(np.nan_to_num(fail), det, dt, G), F_cc=float(np.nanmean(fail[det])), F_cc_cp_upper=cp_upper(int(np.nansum(fail[det])), int(det.sum())),
                           S_failed=float(acc[fu].mean()), E_failed=float((acc & (R <= lead))[fu].mean()), SN_failed=float(((R > lead) & (R <= lead + c['u']))[fu].mean())))
    pd.DataFrame(e3).to_csv(f'{ROOT}/results/{ds}_v2_E3.csv', index=False); pd.DataFrame(e1).to_csv(f'{ROOT}/results/{ds}_v2_E1.csv', index=False)
    pd.set_option('display.width', 250); print(pd.DataFrame(e3).round(4).to_string()); print(pd.DataFrame(e1).round(4).to_string())

if __name__ == '__main__':
    if sys.argv[1] == 'run': run(sys.argv[2], int(sys.argv[3]))
    else: report(sys.argv[2])
