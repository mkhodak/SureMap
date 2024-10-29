import os
import pdb
import warnings
from itertools import product
import numpy as np; np.seterr(all='raise')
from matplotlib import pyplot as plt
import estimators
import suremap
from data import mpi4py_reduce, parse, write, WhisperCVC
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
                  {'theta': 'MetaMap'},
                  {'label': 'MT SureMap', 'color': 'maroon', 'linestyle': 'solid'}),
              ]

def main():

    args, _, comm = parse(default_dataset='WhisperCVC', verbose=False)
    rank, size = comm.rank, comm.size
    alphas = np.round(np.linspace(0., 1., 11), 1)
    rmse = {}
    mae = {}

    for alpha in alphas:

        dataset = WhisperCVC(alpha=alpha,
                             disaggregate_sex='s' in args.disagg,
                             disaggregate_age='a' in args.disagg)
        if not rank:
            np.random.seed(0)
            for task in dataset.tasks:
                dataset.errors(task)
        if size > 1:
            comm.bcast(None)
        if rank:
            for i, task in enumerate(dataset.tasks):
                if i % size == rank:
                    dataset.errors(task)
        write(f'alpha {alpha}: finished loading\r', rank)

        T = len(dataset.tasks)

        rmse[alpha] = np.zeros((args.num_trials, len(ESTIMATORS), len(dataset.group_types)))
        mae[alpha] = np.zeros((args.num_trials, len(ESTIMATORS), len(dataset.group_types)))
        for trial in range(args.num_trials):
            if trial % size != rank:
                continue
            np.random.seed(trial)
            mts = dataset.subsample(args.sample_rate)
            for i, (estimator, kwargs, _) in enumerate(ESTIMATORS):
                estimates = estimator(mts, **kwargs)
                for j, group_type in enumerate(dataset.group_types):
                    rmse[alpha][trial,i,j] = np.sqrt(dataset.mses(estimates, groups=group_type)).mean()
                    mae[alpha][trial,i,j] = dataset.maes(estimates, groups=group_type).mean()
            write(f'alpha {alpha}: finished trial {trial}\r', rank)
        rmse[alpha] = mpi4py_reduce(rmse[alpha], comm)
        mae[alpha] = mpi4py_reduce(mae[alpha], comm)
        write(f'alpha {alpha}: finished evaluating estimators\n', rank)
            
    if (not args.no_plots) and (not rank):
        output_dir = os.path.join(args.output_dir, 'multi', args.dataset)
        os.makedirs(output_dir, exist_ok=True)
        for k, (metric, name) in enumerate([(rmse, 'RMSE'), (mae, 'MAE')]):
            for j, g in enumerate(dataset.group_types):
                for i, (_, _, kwargs) in enumerate(ESTIMATORS):
                    plt.errorbar(alphas,
                                 [np.mean(metric[alpha][:,i,j]) for alpha in alphas],
                                 [2.*np.std(metric[alpha][:,i,j])/np.sqrt(args.num_trials) for alpha in alphas],
                                 capsize=2,
                                 **kwargs)
                plt.legend(fontsize=14)
                plt.xlabel('task similarity', fontsize=18)
                plt.ylabel(f'task-averaged {name}{f"" if g == "all" else f"({g} groups)"}', fontsize=16)
                plt.title(dataset.name, fontsize=18)
                plt.xticks(fontsize=12)
                plt.yticks(fontsize=12)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f'similarity-{args.disagg}_{args.sample_rate}_{g}_{name}.png'), dpi=256)
                plt.clf()

if __name__ == '__main__':
    main()
