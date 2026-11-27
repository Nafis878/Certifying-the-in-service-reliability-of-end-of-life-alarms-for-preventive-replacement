#!/bin/bash
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1
P=${PYTHON:-python}
( for ds in FD002 FD004 PHM08; do for s in 0 1 2; do $P e1_fit.py $ds $s; done; done ) > ../results/log_a.txt 2>&1 &
( for ds in FD001 FD003; do for s in 0 1 2; do $P e1_fit.py $ds $s; done; done; for ds in PHM08 FD001 FD002 FD003 FD004 NCMAPSS6; do $P e2_fit.py $ds; done ) > ../results/log_b.txt 2>&1 &
wait
echo ALLDONE >> ../results/log_b.txt
