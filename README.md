# SureMap

Code to reproduce the results in the NeurIPS 2024 paper *SureMap: Simultaneous mean estimation for single-task and multi-task disaggregated evaluation*.

## Reproducing the results

To reproduce results, clone this repository and run the following commands (tested with Python 3.11.9 on Ubuntu):
```
pip install requirements.txt
apt-get update --yes
apt-get install -y bzip2
bzip2 -d data/common_voice/*  # only needed if evaluating on Common Voice and CVC tasks
bash cost.sh                  # Figure 7
bash multi.sh                 # Figures 3, 12, 13, & 15
bash order.sh                 # Figure 4 (left)
bash similarity.sh            # Figure 4 (right)
bash single.sh                # Figures 2, 8, 9, 10, 11, & 14
bash tasks.sh                 # Figure 5
bash theta.sh                 # Figure 6
```
Note that using the configuration in the Dockerfile is only needed for re-generating the datasets provided in `data/adult-llama.csv` and `data/common_voice`.

## Performing your own disaggregated evaluation

The methods in `suremap.py` assume data provided according to the `MultiTaskStatistics` class in `data.py`. 
To run the single-task method, suppose we have a list `errors` of NumPy arrays and a list `groups` of tuples such that at each index `i` the array `errors[i]` contains the errors of all members of the group encoded by `groups[i]`. 
The entries of the latter encode intersectional groups using tuples of attributes, for example `(race, sex, age)` (avoid using `None` as an attribute class).
Then run the following commands:
```
import suremap
from data import MultiTaskStatistics as MTS
estimate = suremap.single_task(MTS([errors], groups=groups))
```
The above will return a disaggregated evaluation in the form of a NumPy array of size 1 x d, where d is the number of groups.
In the multi-task setting we can instead directly pass a list of lists of NumPy arrays to `MTS`, with the entries of the outer list corresponding to individual tasks.
Then we can use `suremap.multi_task` to output a NumPy array of size T x d, where T is the number of tasks.
To estimate AUC instead of mean error statistics, lists of NumPy arrays of scores and labels can be passed as keyword arguments to `MTS`.

## Citation

```
@inproceedings{khodak2024suremap,
  author={Mikhail Khodak and Lester Mackey and Alexandra Chouldechova and Miroslav Dud\'ik},
  title={SureMap: Simultaneous mean estimation for single-task and multi-task disaggregated evaluation},
  booktitle={Advances in Neural Information Processing Systems},
  year={2024}
}
```
