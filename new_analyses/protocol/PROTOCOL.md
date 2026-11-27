# Pre-specified protocol: experiments E1-E6 for the RESS resubmission

Version 1.0. Written 2026-11-27 (UTC+6) BEFORE any of the analyses below was run. The protocol, the analysis code it
names and the frozen input records are hashed in `SHA256SUMS` before execution. Status: **pre-specified, internally
frozen**. It was not registered on a public registry and the authors waived the approval checkpoint ("execute
everything"); results must therefore be labelled *pre-specified* (E1, E2) or *exploratory* (E3-E6), never
*confirmatory (registered)*.

## Common elements
* Recipe: the published gradient-boosted alarm (eol_window2.py / multi_eval.py, unchanged), environment pinned in
  `requirements.lock` (numpy 2.4.4, pandas 3.0.2, scikit-learn 1.8.0, scipy 1.17.1). Reproduction check before any
  new analysis: FD001 outer fold 1 reproduces the saved per-unit alarms exactly (max |diff| 1.8e-15).
* Engine data sets: PHM08 (218 units), C-MAPSS FD001 (100), FD002 (260), FD003 (100), FD004 (249); N-CMAPSS six
  sensors (99) in E2-E6 only. Acceptance window [-13, +2] cycles (b = 2), warning level H = 20 (task level).
* Not feasible in this workspace and therefore NOT part of this protocol: Severson, HUST, XJTU, TJU and bearing
  re-fits (different pipelines; the HUST/seed-7101 confirmatory per-unit records are not in the workspace).
* alpha = 0.05 throughout. All units are run to failure; R = remaining life at the alarm, Y = reported estimate,
  A = 1 if an alarm fired before failure. Accepted: Y - R in [-13, 2]. Accepted-yet-failed at lead l: accepted and R <= l.
  In-service failure at lead l: no alarm, or R <= l. Discarded life U(l) = mean over units of (R - l)+ for alarmed units.

## E1  Frozen-policy calibration (Theorem 1 applied as stated) - PRE-SPECIFIED
Design per data set and training seed s in {0, 1, 2}: permute units with numpy default_rng(9100 + s); training set =
first round(0.4 n) units; pool = the rest. Fit the recipe on the training set only (inner 5-fold OOF seed 9200 + s,
setting selection exactly as the recipe), freeze (cap, span, k, shift), apply once to every pool unit and store the
per-unit record. No pool outcome is inspected before the policy is frozen. Pool units are split into calibration and test
halves (calibration = first floor(|pool|/2) units of a permutation).
* Primary split: default_rng(9300) on the pool of training seed 0.
* Coverage re-splits: default_rng(9400 + r), r = 0..499, for each training seed (1500 splits per data set).
Calibration: m = number of calibration units with an alarm; T = min_j Y_j - b; q_m = 1 - alpha^(1/m). Certified leads:
integers 0 <= l < T. Logged unit IDs per role: `e1_roles_<ds>_s<s>.csv`.
Expected sizes (seed 0): PHM08 87 train / 65 cal / 66 test; FD002 104/78/78; FD004 100/74/75; FD001 and FD003 40/30/30.
Achievable q_m at 95%: m = 30 -> 9.5%; 66 -> 4.4%; 78 -> 3.8%; 299 needed for 1% (exact values reported).
Hypotheses:
* **H-E1a (primary, per data set, seed 0 primary split):** at every certified lead l, the observed accepted-yet-failed
  share on the test cohort (denominator: all test units) is <= q_m. Met / not met per data set. If T <= 0 no lead is
  certified and H-E1a is reported as "no certified lead" (not met for the purpose of deployment).
* **H-E1b (coverage):** violation indicator per split: pi_hat > q_m, where pi_hat = share of alarmed test units with
  Y < min Y_cal (the event that contains every accepted failure at a certified lead). Under exact exchangeability the
  violation probability with finite test size is pi_0(m, n_t) (exact, hypergeometric order-statistic formula; reported).
  Criterion met if the empirical violation rate over the 1500 splits <= max(alpha, mean pi_0) + 0.03. Splits share units
  and are not independent; no p-value is attached.
* **E1c (descriptive):** share of splits with T > 0; distribution of T and of the largest certified lead; trade-off
  with m by sub-sampling the calibration alarms to m in {10, 20, 40, all} (first m of the calibration order).
* **E1d (registered variant):** k-th order statistic limit, k in {2, 3}: T_k = k-th smallest Y - b; q_{m,k} = smallest
  q with P(Bin(m, q) <= k - 1) <= alpha (bisection). Same H-E1a and H-E1b checks.
* **E1e (comparison):** the out-of-fold training-minimum threshold T_oof = min over the training OOF alarms of Y - b
  (the current paper's retrospective audit) applied to the same test cohort; report its test accepted-failure share.

## E2  Probabilistic-RUL failure-probability policy - PRE-SPECIFIED
Policy (after Mitici et al. 2023 and de Pater et al. 2022: replace when the predicted probability of failure within the
lead time exceeds a threshold). Deviation: the predictive RUL distribution is the recipe's median regressor (cap 90)
plus the empirical distribution of training out-of-fold residuals R - Yhat in 10 bins of Yhat (quantile bins on the
training OOF predictions); no MC dropout or quantile network. At cycle t the policy alarms at the first cycle where
P_hat(R_t <= l + 1) >= p*, p* in {0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50}; p* is chosen per (l, rho) by the training
OOF cost rate (same renewal-reward cost as the paper: 1 per planned, rho per failure, rate = cost / operated cycles).
Splits: the paper's outer 5-fold split (seed 2026) and inner OOF seed 100 + fold, so every policy sees identical units.
Grid: l in {0, 5, 10, 15}, rho in {5, 10, 20, 50}. Comparators: optimal age replacement (age chosen on training lives),
the recipe with its level chosen by training cost (recomputed per unit from the saved records), perfect information.
* **H-E2a:** the probabilistic policy is cheaper than optimal age replacement: pooled mean (over data sets and 16 cells)
  log cost-rate ratio < 0 with the 95% cluster-bootstrap upper bound < 0.
* **H-E2b (non-inferiority, direction fixed in advance):** the recipe is non-inferior to the probabilistic policy:
  upper 95% bound of the pooled mean log(cost_recipe / cost_PP) < +0.05.
Bootstrap: units resampled within data set (B = 2000, seed 9500), all cells recomputed. Also reported per data set:
F(l), U(l), and the mean log ratios.

## E3  Cost and lead-time sensitivity - EXPLORATORY
From the saved per-unit records (seed 2026): rho in {2, 3, 5, 7, 10, 15, 20, 30, 50, 75, 100, 150, 200},
l in {0, 5, 10, 15, 20, 25, 30}, and random lead times L ~ discrete Uniform{0..15} and {0..30} (expected cost over L).
Policies: recipe with level chosen by training cost, recipe at the task level H = 20, optimal age replacement.
Report mean log cost-rate ratios and the break-even rho (smallest grid rho at which the recipe is cheaper than age).

## E4  Dependence-aware equivalence re-analysis - EXPLORATORY (substitute design)
The registered H1/H2 records are not in the workspace. Substitute on the seed-2026 records at H = 20: D = window score
of the recipe minus window score of the training-selected constant-estimate control (constant maximising the training
OOF window score on the same alarms). Compare a unit-independence 90% interval with (i) a cluster bootstrap over
data set x fold clusters and (ii) a DerSimonian-Laird random-effects pooled estimate over data sets; equivalence margin
+-5 points (TOST). Report what changes.

## E5  Censoring simulation for the IPCW estimator (Eq. 19) - SIMULATION, EXPLORATORY
Seed-2026 test records, H = 20. Truth: S on fully observed units. Censoring scenarios: (I) independent C ~ Uniform(
0.5 median life, 1.6 max life); (P) informative: with probability 0.5 the unit is replaced preventively at t* + 15 cycles
after its alarm (censored if this precedes failure), otherwise C as in (I). IPCW with a Kaplan-Meier censoring
survival; naive complete-case estimator for contrast. 1000 replicates per data set and scenario (seed 9600); report bias,
RMSE and coverage of the normal 95% interval (SE from the weighted terms).

## E6  Window-class generality - EXPLORATORY
Same alarms (seed 2026, H = 20, tuned for [-13, 2]) scored under windows [-13,+2], [-5,+5], [-10,+10], [-20,+20],
[-20,+5], [-30,+10], [-2,+13], and a lead-aware window that also requires R > l. Report S_w, accepted-yet-failed
E_w(l) for l = 0..15, F(l) (window-free). Also F(l) for alarms re-tuned to each of the paper's four windows
(saved windows file). Interpretation fixed in advance: if E_w(l) at l = 15 is driven by the upper bound b and F(l) is
unchanged across windows, the finding is reported as structural (a property of the alarm and lead time, not the window).
