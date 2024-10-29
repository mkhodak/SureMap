import os
import pdb
import warnings
import numpy as np; np.seterr(all='raise')
from matplotlib import pyplot as plt
import estimators
import suremap
from data import mpi4py_reduce, parse, write
from sr import structured_regression

            
ESTIMATORS = [
              (estimators.naive_estimator, 
                  {}, 
                  {'label': 'naive', 'color': 'forestgreen', 'linestyle': 'dotted'}),
              (estimators.pooled_estimator, 
                  {}, 
                  {'label': 'pooled', 'color': 'black', 'linestyle': 'dotted'}),
              (estimators.bock_estimator, 
                  {}, 
                  {'label': 'Bock', 'color': 'darkorange', 'linestyle': 'dashed'}),
              (structured_regression,
                  {},
                  {'label': 'Struct. Reg.', 'color': 'olive', 'linestyle': 'dashed'}),
              (suremap.single_task,
                  {},
                  {'label': 'SureMap', 'color': 'indigo', 'linestyle': 'dashed'}),
              ]

def main():

    args, dataset, comm = parse(default_dataset='Diabetes')
    rank, size = comm.rank, comm.size

    rmse = {}
    mae = {}
    for rate in args.sample_rates:
        rmse[rate] = np.zeros((args.num_trials, len(ESTIMATORS), len(dataset.group_types)))
        mae[rate] = np.zeros((args.num_trials, len(ESTIMATORS), len(dataset.group_types)))
        for trial in range(args.num_trials):
            if trial % size != rank:
                continue
            np.random.seed(trial)
            mts = dataset.subsample(rate)
            for i, (estimator, kwargs, _) in enumerate(ESTIMATORS):
                estimates = estimator(mts, **kwargs)
                for j, group_type in enumerate(dataset.group_types):
                    rmse[rate][trial,i,j] = np.sqrt(dataset.mses(estimates, groups=group_type)).mean()
                    mae[rate][trial,i,j] = dataset.maes(estimates, groups=group_type).mean()
            write(f'rate {round(rate, 2)}: finished trial {trial}\r', rank)
        rmse[rate] = mpi4py_reduce(rmse[rate], comm)
        mae[rate] = mpi4py_reduce(mae[rate], comm)
        write(f'rate {round(rate, 2)}: finished evaluating estimators\n', rank)
            
    if not rank:
        output_dir = os.path.join(args.output_dir, 'single', args.dataset)
        os.makedirs(output_dir, exist_ok=True)
        for k, (metric, name) in enumerate([(rmse, 'RMSE'), (mae, 'MAE')]):
            for j, g in enumerate(dataset.group_types):
                for i, (_, _, kwargs) in enumerate(ESTIMATORS):
                    plt.errorbar(args.sample_rates, 
                                 [np.mean(metric[rate][:,i,j]) for rate in args.sample_rates],
                                 [2.*np.std(metric[rate][:,i,j])/np.sqrt(args.num_trials) for rate in args.sample_rates],
                                 capsize=3,
                                 linewidth=3,
                                 **kwargs)
                if j == 1:
                    plt.legend(fontsize=20)
                plt.xlabel('subsampling rate (median group size)', fontsize=18)
                plt.ylabel(f'{name}{f"" if g == "all" else f"({g} groups)"}', fontsize=16)
                plt.title(dataset.name, fontsize=20)
                xticks = []
                ax = plt.gca()
                ax.set_xscale('log')
                for rate in args.sample_rates[::2]:
                    attr = f'{"large_enough" if g == "all" else g}_groups'
                    get_counts = lambda task: dataset.group_counts(task)[getattr(dataset, attr)(task)]
                    group_size = round(np.median([np.median(rate*get_counts(task))
                                                  for task in dataset.tasks]))
                    xticks.append(f'{round(rate, 3)}\n({group_size})')
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    ax.set_xticks(args.sample_rates[::2])
                    ax.set_xticklabels(xticks)
                    plt.xticks(fontsize=12)
                    plt.yticks(fontsize=12)
                    plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f'{args.disagg}_{g}_{name}.png'), dpi=256)
                plt.clf()

if __name__ == '__main__':
    main()
