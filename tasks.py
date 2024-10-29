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
    Ts = np.round((T**.2) ** np.arange(1, 6)).astype(int)
    trials = []
    rmse = {}
    mae = {}

    for num_tasks in Ts:
        num_trials = round(args.num_trials * np.sqrt(T / num_tasks))
        trials.append(num_trials)
        rmse[num_tasks] = np.zeros((num_trials, len(ESTIMATORS), len(dataset.group_types), num_tasks))
        mae[num_tasks] = np.zeros((num_trials, len(ESTIMATORS), len(dataset.group_types), num_tasks))
        for trial in range(num_trials):
            if trial % size != rank:
                continue
            np.random.seed(trial)
            tasks = set(np.random.choice(dataset.tasks, num_tasks, replace=False))
            tasks = [task for task in dataset.tasks if task in tasks]
            mts = dataset.subsample(args.sample_rate, tasks=tasks)
            for i, (estimator, kwargs, _) in enumerate(ESTIMATORS):
                estimates = estimator(mts, **kwargs)
                for j, group_type in enumerate(dataset.group_types):
                    rmse[num_tasks][trial,i,j] = np.sqrt(dataset.mses(estimates, groups=group_type, tasks=tasks))
                    mae[num_tasks][trial,i,j] = dataset.maes(estimates, groups=group_type, tasks=tasks)
            write(f'num_tasks {num_tasks}: finished trial {trial}\r', rank)
        rmse[num_tasks] = mpi4py_reduce(rmse[num_tasks], comm)
        mae[num_tasks] = mpi4py_reduce(mae[num_tasks], comm)
        write(f'num_tasks {num_tasks}: finished evaluating estimators\n', rank)
            
    if (not args.no_plots) and (not rank):
        output_dir = os.path.join(args.output_dir, 'multi', args.dataset)
        os.makedirs(output_dir, exist_ok=True)

        for k, (metric, name) in enumerate([(rmse, 'RMSE'), (mae, 'MAE')]):
            for j, g in enumerate(dataset.group_types):
                for i, (_, _, kwargs) in enumerate(ESTIMATORS):
                    plt.errorbar(Ts, 
                                 [np.mean(metric[num_tasks][:,i,j]) for num_tasks in Ts],
                                 [2.*np.std(metric[num_tasks][:,i,j].mean(1))/np.sqrt(num_trials) 
                                  for num_tasks, num_trials in zip(Ts, trials)],
                                 capsize=2,
                                 **kwargs)
                plt.legend(fontsize=14)
                plt.xlabel('number of tasks', fontsize=18)
                plt.ylabel(f'task-averaged {name}{"" if g == "all" else f" ({g} groups)"}', fontsize=16)
                plt.xlim([1, T+1])
                plt.title(dataset.name, fontsize=18)
                ax = plt.gca()
                xticks = [int(tick) for tick in ax.get_xticks() if int(tick) == tick and tick <= T and tick >= min(Ts)]
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    ax.set_xticks(xticks)
                    ax.set_xticklabels([str(tick) for tick in xticks])
                    plt.xticks(fontsize=12)
                    plt.yticks(fontsize=12)
                    plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f'tasks-{args.disagg}_{args.sample_rate}_{g}_{name}.png'), dpi=256)
                plt.clf()


if __name__ == '__main__':
    main()
