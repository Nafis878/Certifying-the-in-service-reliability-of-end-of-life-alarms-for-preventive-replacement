"""Builds the SCANIA Component X unit-time panel (protocol.md section B2). Unit = vehicle_id. Time = time_step.
Failure = in_study_repair == 1 at length_of_study_time_step (train_tte.csv); otherwise censored at that time step
(end of study for the vehicle; mechanism: administrative/other). Only the training split carries per-vehicle event
times and is used; validation/test labels are class windows, not event times, and are not used."""
import sys, pandas as pd
D = sys.argv[1] if len(sys.argv) > 1 else 'data'
ro = pd.read_csv(f'{D}/scania/train_operational_readouts.csv'); tte = pd.read_csv(f'{D}/scania/train_tte.csv')
spec = pd.read_csv(f'{D}/scania/train_specifications.csv')
U = tte.rename(columns={'vehicle_id': 'unit', 'length_of_study_time_step': 'life', 'in_study_repair': 'failed'})
U['entry'] = 0; U['cens_type'] = U.failed.map({1: 'failure', 0: 'administrative_or_other'})
U = U.merge(spec[['vehicle_id', 'Spec_0']].rename(columns={'vehicle_id': 'unit', 'Spec_0': 'model'}), on='unit', how='left')
U['site'] = 'NA'
P = ro.rename(columns={'vehicle_id': 'unit', 'time_step': 't'})
P.to_parquet(f'{D}/scania_panel.parquet'); U.to_parquet(f'{D}/scania_units.parquet')
print(U.cens_type.value_counts().to_string(), '\nunits', len(U), 'spec groups', U.model.nunique())
