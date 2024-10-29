#!/bin/bash

num_trials=${1:-200}
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
  mpirun -n $num_trials python single.py --disagg $disagg --dataset AdultLlama
  mpirun -n $num_trials python single.py --disagg $disagg --dataset Diabetes
  mpirun -n $num_trials python single.py --disagg $disagg --dataset Diabetes-AUC
  mpirun -n $num_trials python single.py --disagg $disagg --dataset DiabetesRegression
  mpirun -n $num_trials python single.py --disagg $disagg --dataset DiabetesRegression-MSE
done

for disagg in ${SA[@]} ; do
  mpirun -n $num_trials python single.py --disagg $disagg --dataset WhisperCV
  mpirun -n $num_trials python single.py --disagg $disagg --dataset WhisperCV-CER
done
