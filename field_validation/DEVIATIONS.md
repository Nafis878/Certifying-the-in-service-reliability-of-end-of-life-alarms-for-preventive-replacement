# Deviations from protocol.md (logged before the SCANIA operational readouts were downloaded or opened)

Logged 2026-11-27T07:49Z. Only train_tte.csv (event counts) and train_specifications.csv had been opened.

- D1 Implementation gap fixed: protocol B(b) chooses the failure-probability threshold p* by training cost, but
  field.py v1 scored every p*. Now p* is chosen per (lead, rho) on the training out-of-fold alarms of failed training
  units (the same population the recipe's selection uses); E1 reports the threshold chosen for rho = 10.
- D2 SCANIA version: the SND record now serves version 3 (DOI 10.5878/bnh5-ka77) as the latest; the protocol named
  version 2 (DOI 10.5878/jvb5-d390). Version 3 is used; the difference between versions was not checked.
- D3 Failure definition (SCANIA): the data descriptor states that a component that was changed or marked as repaired
  is labelled failed. The event is therefore "repair or replacement of Component X", which may include some planned
  replacements; results are reported with this caveat. Time steps are relative and perturbed by the data owner.
- D4 Download route: data obtained through the authors' own computer (in-browser download, gzip, staged copy);
  SHA-256 of the decompressed files recorded in data_manifest.csv.
- D5 Performance only: fp_alarm caches the residual arrays as numpy arrays (identical outputs); applied after the
  seed-0 policy was frozen and before any calibration or test unit had been scored.
