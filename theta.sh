#!/bin/bash

num_trials=${1:-40}
disagg_all=${2:-false}

echo "evaluating $num_trials random trials"
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
  mpirun -n $num_trials python theta.py --disagg $disagg
done

for disagg in ${SA[@]} ; do
  mpirun -n $num_trials python theta.py --disagg $disagg --dataset WhisperCVC-0.5
done
