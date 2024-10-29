import pdb
import numpy as np; np.seterr(all='raise')


def pooled_estimator(mts, offset=None):
    if offset is None:
        return mts.pooled_estimator
    return ((mts.y - offset) * mts.normalized_precision).sum(1)[:,None] * mts.ones + offset

def naive_estimator(mts):
    output = np.copy(mts.y)
    output[mts.nan] = mts.pooled_estimator[mts.nan]
    return output

def multitask_global(mts, share_sigmas=True):
    w = mts.unscaled_precision if share_sigmas else mts.inverse_variances
    theta = (mts.y * w).sum(0)
    group_totals = w.sum(0)
    theta[group_totals > 0] /= group_totals[group_totals > 0]
    theta[group_totals == 0] = (theta * group_totals).sum() / group_totals.sum()
    return theta[None] * mts.ones

def multitask_offset(mts, **kwargs):
    return pooled_estimator(mts, offset=multitask_global(mts, **kwargs))

def bock_estimator(mts, theta=None, pooled=True, sigma_squared=None):
    output = np.empty((mts.T, mts.d))
    x = np.copy(mts.y)
    if not theta is None:
        x -= theta
    if pooled:
        x -= ((mts.unscaled_precision*x).sum(1) / mts.unscaled_precision.sum(1))[:,None] * mts.ones
        c = mts.deff - 3.
    else:
        c = mts.deff - 2.
    inverse_variances = mts.inverse_variances if sigma_squared is None else mts.unscaled_precision / sigma_squared
    denom = (inverse_variances * x * x).sum(1)
    zeros = denom == 0.
    c[np.logical_and(zeros, c == 0.)] = 1.
    denom[zeros] = c[zeros]
    output[mts.nan] = -x[mts.nan]
    output[mts.notnan] = mts.y[mts.notnan] - (np.clip(c / denom, 0., 1.)[:,None] * x)[mts.notnan]
    return output

def multitask_bock(mts, pooled=False, share_sigmas=True):
    w = mts.unscaled_precision if share_sigmas else mts.inverse_variances
    theta = -mts.y * w
    theta -= theta.sum(0)
    loo_totals = w.sum(0) - w
    gt0 = loo_totals > 0
    theta[~gt0] = ((theta.sum(1) / loo_totals.sum(1))[:,None] * mts.ones)[~gt0]
    theta[gt0] /= loo_totals[gt0]
    sigma_squared = (mts.shared_variance * w.sum(1)).sum() / w.sum() if share_sigmas else None
    return bock_estimator(mts, theta=theta, pooled=pooled, sigma_squared=sigma_squared)


if __name__ == '__main__':

    from data import AdditivePrior, MultiTaskStatistics

    np.random.seed(0)
    dataset = AdditivePrior()
    d, T = dataset.d, dataset.T
    mts = MultiTaskStatistics([dataset.errors(task) for task in dataset.tasks])

    print('Testing pooled estimator')
    truth = np.array([np.concatenate(dataset.errors(task)).mean() * np.ones(d)
                      for task in dataset.tasks])
    print('\terror:', np.linalg.norm(truth - pooled_estimator(mts)))

    print('Testing naive estimator')
    truth = np.array([[group_errors.mean() for group_errors in dataset.errors(task)]
                      for task in dataset.tasks])
    print('\terror:', np.linalg.norm(truth - naive_estimator(mts)))

    print('Testing multi-task pooled estimator')
    truth = np.array([np.concatenate([dataset.errors(task)[i] for task in dataset.tasks]).mean()
                      for i in range(d)])[None,:] * mts.ones
    print('\terror:', np.linalg.norm(truth - multitask_pooled(mts)))

    print('Testing multitask offset')
    theta = np.copy(truth)
    for i, task in enumerate(dataset.tasks):
        counts = dataset.group_counts(task)
        y = np.array([group_errors.mean() for group_errors in dataset.errors(task)])
        truth[i] += ((y-theta[i])*counts).sum() / counts.sum()
    print('\terror:', np.linalg.norm(truth - multitask_offset(mts)))

    print('Testing Bock estimators')
    mxc = mts.means_zero_nans * mts.counts
    idx = np.arange(T)
    for multitask in [False, True]:
        for pooled in [False, True]:
            truth = np.empty((T, d))
            estimator = multitask_bock if multitask else bock_estimator
            if multitask:
                sigma_squared = (mts.shared_variance * mts.counts.sum(1)).sum() / mts.counts.sum()
            for i, task in enumerate(dataset.tasks):
                errors = dataset.errors(task)
                y = np.array([group_errors.mean() for group_errors in errors])
                counts = dataset.group_counts(task)
                if multitask:
                    noti = idx != i
                    x = y - mxc[noti].sum(0) / mts.counts[noti].sum(0)
                else:
                    x = np.copy(y)
                if pooled:
                    x -= (counts * x).sum() / counts.sum()
                    c = d - 3.
                else:
                    c = d - 2.
                inverse_variances = mts.counts[i] * mts.counts.sum() / (mts.shared_variance * mts.counts.sum(1)).sum()
                denom = (x * x * inverse_variances).sum()
                truth[i] = y - np.clip(c / denom, 0., 1.) * x
            print('\terror:', np.linalg.norm(truth - estimator(mts, pooled=pooled)))
