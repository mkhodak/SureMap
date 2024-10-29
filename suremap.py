import pdb
import numpy as np; np.seterr(all='raise')
import scipy as sp
from numba import njit
from data import write


def map_estimator(y, precision, theta, Lambda, cache=None):
    '''
    input:
        y: T x d array
        precision: T x d array
        theta: d array
        Lambda: d x d array
        cache: list or None
    output:
        T x d array
    '''

    if cache:
        return y - cache[-1][:,:,0]
    T, d = y.shape
    matrices = precision[:,None] * Lambda
    output = np.matmul(matrices, y[:,:,None]) + theta[None,:,None]
    diag_idx = np.arange(d)
    matrices[:,diag_idx,diag_idx] += 1.
    for t in range(T):
        output[t] = sp.linalg.lstsq(matrices[t], output[t], overwrite_a=True, overwrite_b=True, check_finite=False, lapack_driver='gelsy')[0]
    return output[:,:,0]
    

@njit("void(f8[:,:,:])", cache=True)
def inplace_batch_inv(Ms):
    for i in range(Ms.shape[0]):
        Ms[i] = np.linalg.inv(Ms[i])


def map_risk_estimator(y, unscaled_precision, scales, theta, Lambda, cache=None):
    '''
    input:
        y: T x d array
        unscaled_precision: T x d array
        scales: T array
        theta: d array
        Lambda: d x d array
        cache: list or None
    output:
        T array
    '''

    T, d = y.shape
    A = unscaled_precision[:,None] * Lambda
    diag_idx = np.arange(d)
    A[:,diag_idx,diag_idx] += 1.
    inplace_batch_inv(A)

    offset = np.matmul(A, y[:,:,None]-theta[None,:,None])
    if not cache is None:
        cache.clear()
        cache.extend([A, offset])
    return d + (np.square(offset[:,:,0]) * unscaled_precision).sum(1) / scales - 2. * np.diagonal(A, axis1=1, axis2=2).sum(1)    


def st_objective(x, y, unscaled_precision, scales, theta, UUTs, UTs, idx, cache):
    return map_risk_estimator(y, unscaled_precision, scales, theta, UUTs.dot(x), cache=cache) 

def st_gradient(x, y, unscaled_precision, scales, theta, UUTs, UTs, idx, cache):

    A, offset = cache
    offset = offset * unscaled_precision[:,:,None]
    a = offset
    b = np.matmul(A.transpose(0, 2, 1), offset)
    Z = np.matmul(unscaled_precision[:,:,None] * A, A).sum(0)

    g = np.empty(len(x))
    g[0] = np.trace(Z) - ((a * b).sum(1)[:,0] / scales).sum()

    ip = ((np.matmul(UTs[None], a) * np.matmul(UTs[None], b)) / scales).sum(0)
    UTZ = UTs.dot(Z)

    start = 0
    for i, stop in enumerate(idx):
        g[i+1] = np.trace(UTs[start:stop].dot(UTZ[start:stop].T)) - ip[start:stop].sum()
        start = stop

    return 2. * g

def single_task(mts, order=float('inf')):
    output = np.empty((mts.T, mts.d))
    UTs, idx = mts.UTs_idx(order=order)
    UUTs = np.stack([np.eye(mts.d)] + [UTs[i:j].T.dot(UTs[i:j]) 
                                       for i, j in zip(np.append(0, idx[:-1]), idx)]).T
    init = np.append(1., np.zeros(len(idx)))
    bounds = [(0., None)] * (1 + len(idx))
    for t in range(mts.T):
        args = (mts.y[t,None], mts.unscaled_precision[t,None], mts.shared_variance[t,None], mts.theta, UUTs, UTs, idx, [])
        try:
            opt = sp.optimize.minimize(st_objective, init, args=args, method='L-BFGS-B', jac=st_gradient, bounds=bounds)
        except np.linalg.LinAlgError:
            write('warning: SureMap encountered a np.linalg.LinAlgError and terminated parameter optimization early\n')
        output[t] = map_estimator(args[0], args[1], mts.theta, None, cache=args[-1])

    return np.clip(output, *mts.bounds)


def meta_estimator(y, unscaled_precision, Lambda, Gamma, cache=None):
    '''
    input:
        y: T x d array
        unscaled_precision: T x d array
        Lambda: d x d array
        Gamma: d x d array
        cache: list or None
    output:
        d array
    '''

    if cache:
        return cache[-2][:,:,0]
    T, d = y.shape
    A = unscaled_precision[:,None] * Lambda
    diag_idx = np.arange(d)
    A[:,diag_idx,diag_idx] += 1.
    inplace_batch_inv(A)
    A = unscaled_precision[:,:,None] * A
    output = np.matmul(A, y[:,:,None])[:,:,0]
    A = np.linalg.inv(Gamma) + A.sum(0)
    return sp.linalg.lstsq(A, output.T, overwrite_a=True, overwrite_b=True, check_finite=False, lapack_driver='gelsy')[0].sum(1)


def meta_risk_estimator(y, unscaled_precision, scale, Lambda, Gamma, cache=None):
    '''
    input:
        y: T x d array
        unscaled_precision: T x d array
        scale: float
        Lambda: d x d array
        Gamma: d x d array
        cache: list or None
    output:
        float
    '''

    T, d = y.shape
    A = unscaled_precision[:,None] * Lambda
    diag_idx = np.arange(d)
    A[:,diag_idx,diag_idx] += 1.
    inplace_batch_inv(A)
    WA = unscaled_precision[:,:,None] * A

    if Gamma is None:
        M = np.matmul(A.transpose(0, 2, 1), WA)
        B = np.linalg.pinv(M.sum(0))
        M = np.matmul(B[None], M)
        WLambda = None
    else:
        WLambda = WA.sum(0)
        B = np.eye(d) + Gamma.dot(WLambda)
        inplace_batch_inv(B[None])
        M = np.matmul(B.dot(Gamma)[None], WA)

    MA = np.matmul(M, A)
    theta = np.matmul(M, y[:,:,None]).sum(0)[None]
    offset = np.matmul(A, y[:,:,None] - theta)
    if not cache is None:
        cache.clear()
        cache.extend([A, WA, WLambda, B, M, MA, theta, offset])

    return (d + (np.square(offset[:,:,0]) * unscaled_precision).sum(1) / scale + 2. * np.diagonal(MA-A, axis1=1, axis2=2).sum(1)).sum()


def mt_objective(x, y, unscaled_precision, scale, UUTs, UTs, idx, cache):
    return st_objective(x, y, unscaled_precision, scale, np.zeros(y.shape[1]), UUTs, UTs, idx, cache).sum()

def mt_gradient(x, y, unscaled_precision, scale, UUTs, UTs, idx, cache):
    return st_gradient(x, y, unscaled_precision, scale, None, UUTs, UTs, idx, cache)

def suresolve_objective(x, y, unscaled_precision, scale, UUTs, UTs, idx, cache):
    return meta_risk_estimator(y, unscaled_precision, scale, UUTs.dot(x), None, cache)

def suresolve_gradient(x, y, unscaled_precision, scale, UUTs, UTs, idx, cache):

    A, WA, WLambda, B, M, MA, theta, offset = cache

    WA2 = np.matmul(WA, A)
    Z = WA2.sum(0) - 2. * np.matmul(WA, MA).sum(0)
    WA2PA = WA2 + WA
    BWA2PA = np.matmul(B[None], WA2PA)
    Z += 2. * np.matmul(np.matmul(WA2PA.sum(0)[None], M) - WA2PA, np.matmul(A, BWA2PA)).sum(0)
    
    a1 = 2. * np.matmul(WA2PA, offset)
    offset = offset * unscaled_precision[:,:,None]
    AToffset = np.matmul(A.transpose(0, 2, 1), offset)
    b1 = 2. * np.matmul(BWA2PA.transpose(0, 2, 1), AToffset.sum(0)[None])
    a2 = AToffset
    b2 = offset

    g = np.empty(len(x))
    g[0] = (2. * (a1 * b1).sum() - (a2 * b2).sum()) / scale + np.trace(Z)
    
    ip = 2. * (np.matmul(UTs[None], a1) * np.matmul(UTs[None], b1)).sum(0)
    ip -= (np.matmul(UTs[None], a2) * np.matmul(UTs[None], b2)).sum(0)
    UTZ = UTs.dot(Z)

    start = 0
    for i, stop in enumerate(idx):
        g[i+1] = ip[start:stop].sum() / scale
        g[i+1] += np.trace(UTs[start:stop].dot(UTZ[start:stop].T))
    
    return 2. * g

def metamap_objective(x, y, unscaled_precision, scale, UUTs, UTs, idx, cache):
    k = 1 + len(idx[0])
    return meta_risk_estimator(y, unscaled_precision, scale, UUTs[0].dot(x[:k]), UUTs[1].dot(x[k:]), cache)

def metamap_gradient(x, y, unscaled_precision, scale, UUTs, UTs, idx, cache):

    A, WA, WLambda, B, M, MA, theta, offset = cache

    offset = offset * unscaled_precision[:,:,None]
    at = offset
    AToffset = np.matmul(A.transpose(0, 2, 1), offset)
    sumAToffset = AToffset.sum(0)
    bt = np.matmul(M.transpose(0, 2, 1), sumAToffset[None]) - AToffset
    au = WLambda.dot(theta[0]) - np.matmul(WA, y[:,:,None]).sum(0)
    bu = np.matmul(B.T, sumAToffset)

    Zt = np.matmul(WA, A).sum(0)
    sumMA = MA.sum(0)
    Zu = (Zt - WLambda.dot(sumMA)).dot(B)
    Zt += np.matmul(WA, np.matmul(sumMA[None]-A, M) - MA).sum(0)

    gt = np.empty(1 + len(idx[0]))
    gu = np.empty(1 + len(idx[1]))
    gt[0] = (at * bt).sum() / scale + np.trace(Zt)
    gu[0] = (au * bu).sum() / scale + np.trace(Zu)

    ipt = (np.matmul(UTs[0][None], at) * np.matmul(UTs[0][None], bt)).sum(0)
    ipu = UTs[1].dot(au) * UTs[1].dot(bu)
    UTZt = UTs[0].dot(Zt)
    UTZu = UTs[1].dot(Zu)

    start = 0
    for i, stop in enumerate(idx[0]):
        gt[i+1] = ipt[start:stop].sum() / scale
        gt[i+1] += np.trace(UTs[0][start:stop].dot(UTZt[start:stop].T))
        start = stop

    start = 0
    for i, stop in enumerate(idx[1]):
        gu[i+1] = ipu[start:stop].sum() / scale
        gu[i+1] += np.trace(UTs[1][start:stop].dot(UTZu[start:stop].T))
        start = stop
    
    return 2. * np.append(gt, gu)

def multi_task(mts, theta='MetaMap', order=float('inf'), meta_order=float('inf')):

    output = np.empty((mts.T, mts.d))
    scale = (mts.shared_variance * mts.unscaled_precision.sum(1)).sum() / mts.unscaled_precision.sum()
    UTs, idx = mts.UTs_idx(order=order)
    UUTs = np.stack([np.eye(mts.d)] + [UTs[i:j].T.dot(UTs[i:j]) 
                                       for i, j in zip(np.append(0, idx[:-1]), idx)]).T
    init = np.append(1., np.zeros(len(idx)))
    bounds = [(0., None)] * (1 + len(idx))

    if theta == 'MetaMap':
        meta_UTs, meta_idx = mts.UTs_idx(order=meta_order)
        meta_UUTs = np.stack([np.eye(mts.d)] + [meta_UTs[i:j].T.dot(meta_UTs[i:j]) 
                                                for i, j in zip(np.append(0, meta_idx[:-1]), meta_idx)]).T
        init = np.append(init, np.append(1., np.zeros(len(meta_idx))))
        bounds += [(0., None)] * (1 + len(meta_idx))
        args = (mts.y-mts.theta, mts.unscaled_precision, scale, (UUTs, meta_UUTs), (UTs, meta_UTs), (idx, meta_idx), [])
        fun = metamap_objective
        jac = metamap_gradient

    elif theta == 'SureSolve':
        args = (mts.y-mts.theta, mts.unscaled_precision, scale, UUTs, UTs, idx, [])
        fun = suresolve_objective
        jac = suresolve_gradient

    elif theta == 'zero':
        args = (mts.y-mts.theta, mts.unscaled_precision, scale, UUTs, UTs, idx, [])
        fun = mt_objective
        jac = mt_gradient

    else:
        raise(NotImplementedError)

    try:
        opt = sp.optimize.minimize(fun, init, args=args, method='L-BFGS-B', jac=jac, bounds=bounds)
    except np.linalg.LinAlgError:
        write('warning: MT SureMap encountered a np.linalg.LinAlgError and terminated parameter optimization early\n')
    return np.clip(map_estimator(mts.y-mts.theta, mts.unscaled_precision, None, None, cache=args[-1]) + mts.theta, *mts.bounds)


if __name__ == '__main__':

    from data import AdditivePrior, MultiTaskStatistics

    np.random.seed(0)
    scale = .5
    dataset = AdditivePrior(T=4096, scale=scale)

    print('testing regular estimator')
    theta = np.random.normal(0, 1, dataset.d)
    Lambda = np.eye(dataset.d) + 1.
    mts = MultiTaskStatistics([dataset.errors(task) for task in dataset.tasks])
    test = map_estimator(mts.means, mts.counts, theta, Lambda)
    invLambda = np.linalg.inv(scale*Lambda)
    invSigma = mts.counts / scale
    truth = np.stack([np.linalg.inv(invLambda+np.diag(invSigma[t])).dot(invLambda.dot(theta)+invSigma[t]*mts.means[t])
                      for t in range(mts.T)])
    print('  MAP error\t', np.linalg.norm(truth - test))
    A = np.stack([np.linalg.inv(invLambda+np.diag(invSigma[t])).dot(invLambda) for t in range(mts.T)])
    truth = np.array([mts.d-2.*np.trace(A[t])+(np.square(A[t].dot(theta-mts.means[t]))*invSigma[t]).sum()
                      for t in range(mts.T)])
    test = map_risk_estimator(mts.means, mts.counts, scale*np.ones(mts.T), theta, Lambda)
    print('  SURE error\t', np.linalg.norm(truth - test))
    print('  convergence')
    for T in [4, 16, 64, 256, 1024, 4096]:
        mts = MultiTaskStatistics([dataset.errors(task) for task in dataset.tasks[:T]])
        map_estimates = map_estimator(mts.means, mts.counts, theta, Lambda)
        sures = map_risk_estimator(mts.means, mts.counts, scale*np.ones(T), theta, Lambda)
        risks = (np.square(dataset.ground_truths[:T] - map_estimates) * mts.counts / scale).sum(1)
        print('\t\t', T, abs((risks-sures).mean()))

    print('testing meta estimator')
    Gamma = np.eye(dataset.d) + .5
    mts = MultiTaskStatistics([dataset.errors(task) for task in dataset.tasks],
                              groups=dataset.groups)
    test = meta_estimator(mts.means, mts.counts, Lambda, Gamma)
    M = np.stack([np.linalg.inv(scale*Lambda + np.diag(scale / mts.counts[t])) for t in range(mts.T)])
    M = np.matmul(np.linalg.inv(np.linalg.inv(scale*Gamma) + M.sum(0))[None], M)
    truth = np.matmul(M, mts.means[:,:,None])[:,:,0].sum(0)
    print('  MAP error\t', np.linalg.norm(truth - test))
    test = meta_risk_estimator(mts.means, mts.counts, scale*np.ones(mts.T), Lambda, Gamma)
    truth = sum(mts.d+2.*np.trace(A[t].dot(M[t])-A[t])+(np.square(A[t].dot(truth-mts.means[t]))*invSigma[t]).sum()
                for t in range(mts.T))
    print('  SURE error\t', abs(truth - test))
    print('  convergence')
    for T in [4, 16, 64, 256, 1024, 4096]:
        mts = MultiTaskStatistics([dataset.errors(task) for task in dataset.tasks[:T]])
        theta = meta_estimator(mts.means, mts.counts, Lambda, Gamma)
        map_estimates = map_estimator(mts.means, mts.counts, theta, Lambda)
        sure = meta_risk_estimator(mts.means, mts.counts, scale*np.ones(mts.T), Lambda, Gamma)
        risks = (np.square(dataset.ground_truths[:T] - map_estimates) * mts.counts / scale).sum(1)
        print('\t\t', T, abs(sure-risks.sum()) / T)
