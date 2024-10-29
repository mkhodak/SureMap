import os
import pdb
import warnings
from itertools import product
import numpy as np; np.seterr(all='raise')
from matplotlib import pyplot as plt
import estimators
import suremap
from data import mpi4py_reduce, parse, write


ESTIMATORS = [
              (estimators.naive_estimator, 
                  {}, 
                  {'label': 'naive', 'color': 'forestgreen', 'linestyle': 'dotted'}),
              (estimators.pooled_estimator, 
                  {}, 
                  {'label': 'pooled', 'color': 'black', 'linestyle': 'dotted'}),
              (suremap.single_task,
                  {'order': -1},
                  {'label': 'SureMap (-1)', 'color': 'hotpink', 'linestyle': 'dashed'}),
              (suremap.single_task,
                  {'order': 0},
                  {'label': 'SureMap (0)', 'color': 'orchid', 'linestyle': 'dashed'}),
              (suremap.single_task,
                  {'order': 1},
                  {'label': 'SureMap (1)', 'color': 'purple', 'linestyle': 'dashed'}),
              (suremap.single_task,
                  {'order': 2},
                  {'label': 'SureMap (2)', 'color': 'indigo', 'linestyle': 'dashed'}),
              (estimators.multitask_global,
                  {},
                  {'label': 'MT global', 'color': 'black', 'linestyle': 'solid'}),
              (suremap.multi_task,
                  {'order': -1, 'meta_order': -1},
                  {'label': 'MT SureMap (-1)', 'color': 'palevioletred', 'linestyle': 'solid'}),
              (suremap.multi_task,
                  {'order': 0, 'meta_order': 0},
                  {'label': 'MT SureMap (0)', 'color': 'lightcoral', 'linestyle': 'solid'}),
              (suremap.multi_task,
                  {'order': 1, 'meta_order': 1},
                  {'label': 'MT SureMap (1)', 'color': 'indianred', 'linestyle': 'solid'}),
              (suremap.multi_task,
                  {'order': 2, 'meta_order': 2},
                  {'label': 'MT SureMap (2)', 'color': 'maroon', 'linestyle': 'solid'}),
              ]

def main():

    args, dataset, comm = parse()
    rank, size = comm.rank, comm.size
    
    estimators = ESTIMATORS[:3+len(args.disagg)]
    multi = len(dataset.tasks) > 1
    if multi:
        estimators += ESTIMATORS[6:8+len(args.disagg)]

    rmse = {}
    mae = {}
    for rate in args.sample_rates:
        rmse[rate] = np.zeros((args.num_trials, len(estimators), len(dataset.group_types)))
        mae[rate] = np.zeros((args.num_trials, len(estimators), len(dataset.group_types)))
        for trial in range(args.num_trials):
            if trial % size != rank:
                continue
            np.random.seed(trial)
            mts = dataset.subsample(rate)
            for i, (estimator, kwargs, _) in enumerate(estimators):
                estimates = estimator(mts, **kwargs)
                for j, group_type in enumerate(dataset.group_types):
                    rmse[rate][trial,i,j] = np.sqrt(dataset.mses(estimates, groups=group_type)).mean()
                    mae[rate][trial,i,j] = dataset.maes(estimates, groups=group_type).mean()
            write(f'rate {round(rate, 2)}: finished trial {trial}\r', rank)
        rmse[rate] = mpi4py_reduce(rmse[rate], comm)
        mae[rate] = mpi4py_reduce(mae[rate], comm)
        write(f'rate {round(rate, 2)}: finished evaluating estimators\n', rank)
            
    if (not args.no_plots) and (not rank):
        output_dir = os.path.join(args.output_dir, 'multi' if multi else 'single', args.dataset)
        os.makedirs(output_dir, exist_ok=True)
        center = np.empty(dataset.num_groups)
        for i, gt in enumerate(dataset.ground_truths.T):
            notnan = ~np.isnan(gt)
            center[i] = np.median(gt) if notnan.any() else float('nan')
        center[np.isnan(center)] = center[~np.isnan(center)].mean()
        diffs = center - dataset.ground_truths

        for k, (metric, name) in enumerate([(rmse, 'RMSE'), (mae, 'MAE')]):
            for j, g in enumerate(dataset.group_types):
                attr = f'{"large_enough" if g == "all" else g}_groups'
                get_counts = lambda task: dataset.group_counts(task)[getattr(dataset, attr)(task)]
                for i, (_, _, kwargs) in enumerate(estimators):
                    plt.errorbar(args.sample_rates, 
                                 [np.mean(metric[rate][:,i,j]) for rate in args.sample_rates],
                                 [2.*np.std(metric[rate][:,i,j])/np.sqrt(args.num_trials) for rate in args.sample_rates],
                                 capsize=3-multi, 
                                 linewidth=3-multi,
                                 **kwargs)
                plt.legend(fontsize=20-(5+len(args.disagg))*multi)
                plt.xlabel('subsampling rate (median group size)', fontsize=18)
                plt.ylabel(f'{"task-averaged " if multi else ""}{name}{f"" if g == "all" else f"({g} groups)"}', fontsize=16)
                plt.title(dataset.name, fontsize=18)
                xticks = []
                ax = plt.gca()
                ax.set_xscale('log')
                for rate in args.sample_rates[::2]:
                    group_size = round(np.median([np.median(rate*get_counts(task))
                                                  for task in dataset.tasks]))
                    xticks.append(f'{round(rate, 3)}\n({group_size})')
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    ax.set_xticks(args.sample_rates[::2])
                    ax.set_xticklabels(xticks)
                    plt.xticks(fontsize=12)
                    plt.yticks(fontsize=12)
                    plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f'order-{args.disagg}_{g}_{name}.png'), dpi=256)
                plt.clf()


if __name__ == '__main__':
    main()
