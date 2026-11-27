# Field protocol v2 (written 2026-11-27 AFTER the v1 SCANIA results were seen; not blind to v1)

Why: under v1 the simulator-tuned RUL-threshold recipe gave notice before almost no repairs (RESULTS_SCANIA.md).
v2 adds one policy designed for censored fleet data. Everything else in v1 is unchanged: data, seeds, unit splits
and roles, window [-13, +2], b = 2, alpha = 0.05, leads {0, 1, 3, 7, 14, 30}, Algorithm 1, E1 and E3 endpoints and
decision rules. Status label of every v2 result: "pre-specified v2 (post v1)". The v1 results are kept and reported.

Policy (d) hazard alarm:
* Classifier: HistGradientBoostingClassifier (log loss, max_iter 300, learning rate 0.05, random_state = seed) on the
  readouts of training vehicles, features as in v1 plus t. Label 1 if a repair occurs within (t, t + 20]; 0 if the
  vehicle is observed beyond t + 20 without repair; rows whose label is unknown (censored within 20 steps) are dropped.
* Alarm: first readout with predicted probability >= theta, theta in {0.05, 0.1, 0.2, 0.3, 0.4, 0.5}, chosen on the
  training vehicles' out-of-fold predictions (5 folds, seed 1000 + seed) to minimise the IPCW renewal cost rate at
  rho = 10 and lead 7 (middle of the grid).
* Reported estimate Y at the alarm: the frozen v1 recipe regressor's RUL estimate (same seed) at that readout, so that
  window acceptance and Algorithm 1 apply unchanged.
* Additional registered variant: k-th order statistic limits k = 2, 3 (as in the main study's E1d).
