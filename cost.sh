#!/bin/bash

num_trials=${1:-20}
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
  python cost.py --disagg $disagg --num-trials $num_trials
done

for disagg in ${SA[@]} ; do
  python cost.py --dataset WhisperCVC-0.5 --disagg $disagg --num-trials $num_trials
done
