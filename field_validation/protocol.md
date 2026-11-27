# Field-validation protocol (pre-specified; written before any field data were obtained)

Status: written and hashed on 2026-11-27, before any field record was downloaded or seen (the analysis workspace has
no network route to the data hosts). Nothing below has been run on field data. All definitions are the manuscript's:
Algorithm 1 (T = min Y - b, q_m = 1 - alpha^(1/m)), Eqs. (1)-(2), (6), (10)-(13), Propositions 1-2, Theorems 1-2.

## A. Data
B1 Backblaze Drive Stats, quarterly ZIPs Q1 2023 - Q4 2025 (daily SMART snapshots; Backblaze asks for citation,
   responsibility for use and no resale). Unit = serial_number; cohort = drives first seen on or after 2023-02-01
   (new installs, no left truncation); six most numerous models by unit count (outcome-free); time unit = day.
   Failure = failure flag on the last record. Censoring: administrative (still reporting at window end) or removal
   without failure (mechanism unknown; reported separately). Known threat: Backblaze's failure flag may include drives
   pulled on SMART evidence (failure partly alarm-defined) - check against Backblaze's published definition before
   analysis; if it cannot be defended, B1 is rejected.
B2 SCANIA Component X (Kharazian et al., Sci Data 2025; SND-ID 2024-34 v2, DOI 10.5878/jvb5-d390). Unit = vehicle;
   time = time_step; failure = in_study_repair = 1 at length_of_study_time_step (a repair, used as the end-of-life
   event); otherwise censored (end of study). Only the training split carries event times. Transport groups = Spec_0.
   Column names must be checked against the downloaded files.

## B. Policies (training units only; frozen and hashed before any calibration unit is scored)
(a) reference recipe: absolute-error gradient boosting on capped RUL, EWM smoothing, k-consecutive debounce, shift,
    task level H = 20, window [-13, +2] in the dataset's time unit, selection by training out-of-fold window score on
    failed training units; (b) failure-probability policy (empirical out-of-fold residual distribution, threshold p* by
    training cost); (c1) naive first crossing (k = 1, no smoothing); (c2, Backblaze only) SMART raw-counter rule on
    attributes 5, 187, 188, 197, 198. Literature support for (c1)-(c2) must be verified and cited before use.
Splits: by unit, stratified by model group: 40% train, 30% calibration, 30% test; seeds 0, 1, 2.

## C. Endpoints and decision rules
E1 (pre-specified) test-set S (failed units), F(l) by IPCW with model-cluster bootstrap 95% CI and the complete-case
   Clopper-Pearson upper bound, accepted-yet-failed share, S_N(l, u = 15), cost vs optimal age replacement (KM-based),
   rho in {5, 10, 20, 50}, leads {0, 1, 3, 7, 14, 30}. The gap claim is supported if, for the recipe, the
   accepted-yet-failed share at some lead is > 0 on at least one dataset; contradicted if it is 0 at all leads while
   S >= 0.5.
E3 (pre-specified, primary) Algorithm 1 on warned calibration units. Requirement m >= 299; if not met, report the
   achieved q_m. Coverage over 1000 re-splits, unit-level and model-cluster-level; criterion: violation rate <= alpha
   + 0.03 for the cluster design (both reported). Accepted-yet-failed on test at the certified leads (IPCW).
   Total-failure bound (section D) is exploratory until its censored-data version is derived.
E4 (exploratory) held-out model groups; eta_hat = exceedance - q_m.
E5 (exploratory) Kendall tau_b of S and S_N with realised cost across policies.
E6 (exploratory) failure definitions (Backblaze: failure vs failure-or-removal), windows, seeds 0-2, rho grid.
If a pre-specified outcome contradicts the manuscript's claim, analysis stops and the authors are informed first.

## D. Total failure bound (Proposition F1, uncensored calibration)
Let A be the alarm-before-failure indicator and R the notice. F(l) = P(A = 0) + P(A = 1) P(R <= l | A = 1).
With R_1..R_m the notices of m warned calibration units (independent, deployment law) and T_R = min_j R_j:
P(F_R(T_R-) > q) <= (1 - q)^m (probability integral transform with F(x-)), so with q_m = 1 - alpha^(1/m),
P(R <= l | A = 1) <= q_m for every l < T_R, with probability >= 1 - alpha. With p_bar the one-sided (1 - alpha)
Clopper-Pearson upper bound on the miss probability from n calibration units, a union bound gives, with probability
>= 1 - 2 alpha, F(l) <= p_bar + q_m for all l < T_R. This bound is not dominated by 1 - S. Under right censoring R and
the miss indicator are not observed for every unit: [NEEDS DERIVATION] (candidate: treat censored warned units as
failures at their censoring time, which is conservative).
