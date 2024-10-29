#!/bin/bash

num_trials_single=${1:-200}
num_trials_multi=${2:-40}
disagg_all=${3:-false}

echo "evaluating $num_trials_single random single-task trials"
echo "evaluating $num_trials_multi random multi-task trials"
if $disagg_all ; then
  echo "running multiple levels of disaggregations"
  RSA=("r" "rs" "ra" "sa" "rsa")
  SA=("a" "sa")
else
  RSA="rsa"
  SA="sa"
  echo "running only all-attribute disaggregations"
fi

for disagg in ${RSA[@]} ; do
  mpirun -n $num_trials_single python order.py --disagg $disagg --dataset Diabetes
  mpirun -n $num_trials_multi python order.py --disagg $disagg
done

for disagg in ${SA[@]} ; do
  mpirun -n $num_trials_single python order.py --disagg $disagg --dataset WhisperCV
  mpirun -n $num_trials_multi python order.py --disagg $disagg --dataset WhisperCVC-0.5
done
