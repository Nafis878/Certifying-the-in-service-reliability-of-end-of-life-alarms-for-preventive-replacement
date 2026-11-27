"""E2 stage 1: per outer fold (seed 2026, inner seed 100+fold), recipe median regressor (cap 90) predictions for
training-OOF and test units (every cycle), saved for the probabilistic policy."""
import sys, pickle; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from recipe import *
W.CAPS = [90]   # only the cap used by the probabilistic policy is fitted
ds = sys.argv[1]; OUT = os.environ.get('OUT_DIR', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results'))
tr, pred = load(ds); units = np.sort(tr.unit.unique()); life = tr.groupby('unit').cycle.max(); res = {}
for fold, test_u in enumerate(outer_folds(units), 1):
    train_u = np.setdiff1d(units, test_u); inner = np.array_split(np.random.default_rng(100 + fold).permutation(train_u), 5); oof = {}
    for iu in inner: oof.update(pred(np.setdiff1d(train_u, iu), iu)[90])
    res[fold] = dict(train_u=train_u, test_u=test_u, oof=oof, test=pred(train_u, test_u)[90])
    print(ds, fold, flush=True)
pickle.dump(dict(res=res, life=life.to_dict()), open(f'{OUT}/e2_preds_{ds}.pkl', 'wb'))
