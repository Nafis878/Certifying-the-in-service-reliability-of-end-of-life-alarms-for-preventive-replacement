"""Thin wrapper around the published recipe (eol_window2 / multi_eval) so that a policy can be fitted on an
arbitrary training set, frozen, and applied to any other units. No change to the recipe itself."""
import os, sys, numpy as np, pandas as pd
REPO = os.environ.get('STUDY_DIR', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'study', 'code'))
os.chdir(REPO); sys.path.insert(0, REPO)
_argv = sys.argv; sys.argv = sys.argv[:1]
import eol_window2 as W
import multi_eval as ME
sys.argv = _argv
from safe_common import load_dataset, outer_folds

def load(ds):
    if ds.startswith('NCMAPSS'):
        tr, sensors = ME.ncm_frame(ME.NCM6)
        pred = lambda a, b: ME.predict_generic(tr, sensors, a, b)
    else:
        tr, _, _ = load_dataset(ds)
        pred = lambda a, b: W.predict_all(tr, a, b, a)
    return tr, pred

def fit_policy(pred, train_u, H, lo, hi, inner_seed):
    """Recipe: inner 5-fold OOF on training units, select (cap, span, k, shift) maximising the training window score."""
    inner = np.array_split(np.random.default_rng(inner_seed).permutation(train_u), 5); oof = {c: {} for c in W.CAPS}
    for iu in inner:
        p = pred(np.setdiff1d(train_u, iu), iu)
        for c in W.CAPS: oof[c].update(p[c])
    best = None
    for cap in W.CAPS:
        if cap < H + 20: continue
        for sp in W.SPANS:
            for k in W.KS:
                Q = ME.prep({u: (W.smooth(p, sp), y) for u, (p, y) in oof[cap].items()}, k)
                tot = ME.in_counts(Q, H, lo, hi) - 0.001 * np.abs(W.SHIFTS); j = int(np.argmax(tot))
                if best is None or tot[j] > best[0]: best = (tot[j], cap, sp, k, float(W.SHIFTS[j]))
    return dict(cap=best[1], span=best[2], k=best[3], shift=best[4], H=H), oof

def apply_policy(preds_cap, pol):
    """preds_cap: {unit: (p, y)} for the policy's cap. Returns {unit: (pred_at_alarm, rul_at_alarm, alarm_cycle)}"""
    return ME.warnings({u: (W.smooth(p, pol['span']), y) for u, (p, y) in preds_cap.items()}, pol['H'], pol['k'], pol['shift'])
