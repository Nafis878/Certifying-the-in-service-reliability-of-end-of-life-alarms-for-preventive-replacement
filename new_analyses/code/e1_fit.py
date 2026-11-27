"""E1 stage 1: fit and freeze one recipe policy per (data set, training seed); store alarms of pool and training-OOF units.
Outcomes of pool units are written but not inspected here."""
import sys, json; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from recipe import *
OUT = os.environ.get('OUT_DIR', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results')); os.makedirs(OUT, exist_ok=True)
ds, s = sys.argv[1], int(sys.argv[2])
tr, pred = load(ds); units = np.sort(tr.unit.unique()); life = tr.groupby('unit').cycle.max()
perm = np.random.default_rng(9100 + s).permutation(units); ntr = int(round(0.4 * len(units)))
train_u, pool_u = np.sort(perm[:ntr]), np.sort(perm[ntr:])
pol, oof = fit_policy(pred, train_u, 20, -13, 2, 9200 + s)
fin = pred(train_u, pool_u)[pol['cap']]
rows = []
for role, P in (('pool', apply_policy(fin, pol)), ('train_oof', apply_policy(oof[pol['cap']], pol))):
    for u, (y, r, c) in P.items(): rows.append(dict(ds=ds, seed=s, role=role, unit=u, Y=y, R=r, cycle=c, life=life[u]))
pd.DataFrame(rows).to_csv(f'{OUT}/e1_records_{ds}_s{s}.csv', index=False)
json.dump(pol, open(f'{OUT}/e1_policy_{ds}_s{s}.json', 'w'))
print(ds, s, pol, flush=True)
