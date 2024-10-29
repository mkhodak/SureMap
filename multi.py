import os
import pdb
import warnings
from itertools import product
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
              (estimators.multitask_global,
                  {},
                  {'label': 'MT global', 'color': 'black', 'linestyle': 'solid'}),
              (estimators.multitask_offset,
                  {},
                  {'label': 'MT offset', 'color': 'goldenrod', 'linestyle': 'solid'}),
              (estimators.multitask_bock,
                  {},
                  {'label': 'MT Bock', 'color': 'cornflowerblue', 'linestyle': 'solid'}),
              (suremap.multi_task,
                  {},
                  {'label': 'MT SureMap', 'color': 'maroon', 'linestyle': 'solid'}),
              ]

def main():

    args, dataset, comm = parse()
    rank, size = comm.rank, comm.size
    
    T = len(dataset.tasks)
    rmse = {}
    mae = {}
    for rate in args.sample_rates:
        rmse[rate] = np.zeros((args.num_trials, len(ESTIMATORS), len(dataset.group_types), T))
        mae[rate] = np.zeros((args.num_trials, len(ESTIMATORS), len(dataset.group_types), T))
        for trial in range(args.num_trials):
            if trial % size != rank:
                continue
            np.random.seed(trial)
            mts = dataset.subsample(rate)
            for i, (estimator, kwargs, _) in enumerate(ESTIMATORS):
                estimates = estimator(mts, **kwargs)
                for j, group_type in enumerate(dataset.group_types):
                    rmse[rate][trial,i,j] = np.sqrt(dataset.mses(estimates, groups=group_type))
                    mae[rate][trial,i,j] = dataset.maes(estimates, groups=group_type)
            write(f'rate {round(rate, 2)}: finished trial {trial}\r', rank)
        rmse[rate] = mpi4py_reduce(rmse[rate], comm)
        mae[rate] = mpi4py_reduce(mae[rate], comm)
        write(f'rate {round(rate, 2)}: finished evaluating estimators\n', rank)
            
    if (not args.no_plots) and (not rank):
        output_dir = os.path.join(args.output_dir, 'multi', args.dataset)
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
                for i, (_, _, kwargs) in enumerate(ESTIMATORS):
                    plt.errorbar(args.sample_rates, 
                                 [np.mean(metric[rate][:,i,j]) for rate in args.sample_rates],
                                 [2.*np.std(metric[rate][:,i,j].mean(1))/np.sqrt(args.num_trials) for rate in args.sample_rates],
                                 capsize=2,
                                 **kwargs)
                plt.legend(fontsize=14)
                plt.xlabel('subsampling rate (median group size)', fontsize=18)
                plt.ylabel(f'task-averaged {name}{f"" if g == "all" else f"({g} groups)"}', fontsize=16)
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
                plt.savefig(os.path.join(output_dir, f'{args.disagg}_{g}_{name}.png'), dpi=256)
                plt.clf()

                rate = args.sample_rates[len(args.sample_rates) // 2]
                group_size = round(np.median([np.median(rate*get_counts(task))
                                              for task in dataset.tasks]))
                distances = np.array([np.linalg.norm(diffs[t,getattr(dataset, attr)(task)], 
                                                     1 if name == 'MAE' else 2)
                                      for t, task in enumerate(dataset.tasks)])
                counts = np.array([get_counts(task).sum() for task in dataset.tasks])
                for i, shape in [(-2, 'x'), (-1, 'o')]:
                    plt.scatter(distances, 
                            np.mean(metric[rate][:,0,j]/metric[rate][:,i,j], 0),
                                label=ESTIMATORS[i][2]['label'],
                                color=ESTIMATORS[i][2]['color'],
                                marker=shape)
                xlim = plt.xlim()
                plt.plot(xlim, [1., 1.], **ESTIMATORS[0][2])
                plt.xlim(xlim)
                plt.legend(fontsize=14)
                plt.xlabel("distance from ground truth to MT median", fontsize=18)
                plt.ylabel(f"multiplicative improvement\nin {name} over {ESTIMATORS[0][2]['label']}", fontsize=16)
                plt.title(f'task-level results (median grp. size = {group_size})', fontsize=18)
                plt.xticks(fontsize=12)
                plt.yticks(fontsize=12)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f'improvement-{args.disagg}_{round(rate, 2)}_{g}_{name}.png'), dpi=256)
                plt.clf()


if __name__ == '__main__':
    main()
