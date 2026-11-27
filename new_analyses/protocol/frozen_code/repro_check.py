import sys; sys.path.insert(0, '<WORKDIR>')
from recipe import *
tr, pred = load('FD001'); units = np.sort(tr.unit.unique()); test_u = outer_folds(units)[0]; train_u = np.setdiff1d(units, test_u)
pol, _ = fit_policy(pred, train_u, 20, -13, 2, 101)
fin = pred(train_u, test_u); te = apply_policy(fin[pol['cap']], pol)
S = pd.read_csv('results/multi/FD001_warnings.csv'); S = S[(S.fold == 1) & (S.role == 'test') & (S.H == 20)].set_index('unit')
d = [abs(te[u][0] - S.loc[u, 'pred']) for u in test_u if not np.isnan(te[u][0])]
print(pol, 'max abs diff vs saved:', max(d), 'cycles equal:', all(te[u][2] == S.loc[u, 'cycle'] for u in test_u if not np.isnan(te[u][0])))
