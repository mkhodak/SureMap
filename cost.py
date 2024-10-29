import os
import pdb
import time
import warnings
import numpy as np; np.seterr(all='raise')
from matplotlib import pyplot as plt
import estimators
import suremap
from data import parse, write
from data import MultiTaskStatistics as MTS
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

    args, dataset, _ = parse()
    
    T = len(dataset.tasks)
    Ts = np.round((T**.2) ** np.arange(1, 6)).astype(int)
    costs = {}

    for num_tasks in Ts:
        costs[num_tasks] = np.zeros((args.num_trials, len(ESTIMATORS)))
        for trial in range(args.num_trials):
            np.random.seed(trial)
            tasks = set(np.random.choice(dataset.tasks, num_tasks, replace=False))
            tasks = [task for task in dataset.tasks if task in tasks]
            mts = dataset.subsample(args.sample_rate, tasks=tasks)
            if trial == 0:
                for i, (estimator, kwargs, _) in enumerate(ESTIMATORS):
                    estimates = estimator(mts, **kwargs)
            for i, (estimator, kwargs, _) in enumerate(ESTIMATORS):
                costs[num_tasks][trial,i] = -time.perf_counter()
                estimates = estimator(MTS(mts.errors, groups=dataset.groups), **kwargs)
                costs[num_tasks][trial,i] += time.perf_counter()
            write(f'num_tasks {num_tasks}: finished trial {trial}\r')
        write(f'num_tasks {num_tasks}: finished evaluating estimators\n')
            
    if not args.no_plots:
        output_dir = os.path.join(args.output_dir, 'multi', args.dataset)
        os.makedirs(output_dir, exist_ok=True)

        for i, (_, _, kwargs) in enumerate(ESTIMATORS):
            plt.errorbar(Ts, 
                         [np.mean(costs[num_tasks][:,i]) for num_tasks in Ts],
                         [2.*np.std(costs[num_tasks][:,i])/np.sqrt(args.num_trials) 
                          for num_tasks in Ts],
                         capsize=2,
                         **kwargs)
        plt.legend(fontsize=14, bbox_to_anchor=(1.05, 1.0))
        plt.xlabel('number of tasks', fontsize=18)
        plt.ylabel(f'cost (seconds)', fontsize=16)
        plt.xscale('log')
        plt.yscale('log')
        plt.xlim([0.9*min(Ts), 1.1*max(Ts)])
        plt.minorticks_off()
        plt.title(dataset.name[:dataset.name.index('(')-1], fontsize=18)
        ax = plt.gca()
        xticks = Ts
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ax.set_xticks(xticks)
            ax.set_xticklabels([str(tick) for tick in xticks])
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'cost-{args.disagg}.png'), dpi=256)
        plt.clf()


if __name__ == '__main__':
    main()
