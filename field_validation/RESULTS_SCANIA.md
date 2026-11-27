# SCANIA Component X: pre-specified field run (E1, E3). Status: STOPPED at checkpoint (hard rule 6)

Data: SND 2024-34 version 3, training split, 23,550 vehicles; 2,272 repairs (9.6%); 90.4% censored at end of study.
The event is "Component X changed or marked repaired" (data descriptor); time steps are relative and perturbed.
Splits: by vehicle, stratified by Spec_0 (three groups: 19,221 / 4,291 / 38 vehicles), 40/30/30; seeds 0-2.
Frozen policy hashes are in results/policy_hashes.csv. Deviations D1-D5 are listed in DEVIATIONS.md.

## E3: certification (pre-specified, primary)
| Seed | Warned calibration units m (of ~7,065) | T (time steps) | q_m | Meets m >= 299 |
|---|---|---|---|---|
| 0 | 48 | 4.0 | 6.1% | no |
| 1 | 0 | none | none | no |
| 2 | 11 | 11.4 | 23.8% | no |
Coverage (unit re-splits): violation rate 8.7% (seed 0), 0% (seed 1), 9.9% (seed 2); criterion not assessable with these m.
The cluster design has only three Spec_0 groups and is degenerate (0% for every seed).

## E1: field gap (pre-specified)
* Reference recipe: alarmed 86, 1 and 18 of 14,131 non-training vehicles (seeds 0-2), and gave notice before only 9, 0
  and 1 of about 1,330 repairs. Window score on repaired vehicles was 0.0-0.1%. The accepted-yet-failed share was 0
  except 1 vehicle at a 30-step lead (seed 0). IPCW in-service failure probability 0.26-0.31 at every lead, which is
  essentially the repair probability without an alarm.
* Naive first crossing: the same pattern (alarm before at most 3 repairs).
* Failure-probability policy (p* chosen on training cost, rho = 10): IPCW F(l) 0.15-0.17 for leads 0-14; at l = 30 the
  chosen threshold (0.01) replaces every vehicle at its first readout (F = 0, trivial policy).
* Pre-specified decision rule: the gap claim is "supported" only by a single vehicle (seed 0, l = 30) and is not
  "contradicted" (that requires S >= 0.5). **Verdict: inconclusive.** On this fleet the reference alarm almost never
  fires before a repair, so acceptance is almost never granted and the gap cannot be measured.

## What this means
The simulator-tuned recipe does not transfer to censored fleet data with this failure definition: it misses about
95-100% of repairs. Certification at scale (m >= 299) was impossible with the pre-specified policy. This is a negative
result for the policy, not for Proposition 1 or Theorem 1. Nothing here supports adding a "field validation" claim to
the manuscript.

# Protocol v2 (written after v1, hashed before running): censoring-aware hazard alarm
Hashes: protocol_v2.sha256 (written 2026-11-27T08:52Z); frozen v2 policies in results/policy_hashes.csv; theta = 0.05
was selected on training data in every seed.

E3 (k = 1): warned calibration trucks m = 1,140 / 1,147 / 1,120 (requirement m >= 299 met); q_m = 0.26-0.27%;
T = 9.0 / 27.0 / 22.0 time steps; test share of alarmed trucks below the calibration minimum 0 / 0 / 0.17% (<= q_m);
accepted-yet-failed at the largest certified lead: 0 in every seed. Coverage over 1,000 unit re-splits: 9.7%, 9.1%,
10.0%. The pre-set criterion (<= alpha + 0.03 = 8%) is NOT met; the rate expected under exact exchangeability for these
cohort sizes is 12.5-12.7% (computed post hoc, not pre-specified). k = 2, 3: T up to 29.6 steps, q <= 0.56%.
E1: the hazard alarm warned 1,093-1,197 of 7,067 test trucks and gave notice before 172-192 of 661-703 repairs
(25-29%); IPCW in-service failure probability 0.17-0.26 across leads 0-30; window score on repaired trucks 1.8-2.7%;
accepted-yet-failed 0 except 0.15% at l = 30 (seed 0); notice-window score S_N(l, 15) 3-7%.
Reading: certification at scale works and holds on held-out trucks, but the certified alarm is not a reliable one -
it misses most repairs. The certificate and F(l) answer different questions.
