import pdb
import warnings
import numpy as np; np.seterr(all='raise')
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNet
from sklearn.utils import check_random_state

EPS = 1e-6
FOLDS = 10
FRAC = 0.9
LASSO_KWARGS = {
                'l1_ratio': 0.95,
                'selection': 'random',
                'max_iter': 10000,
                'random_state': 33
                }


def rnd_round(x, random_state = None):
    random_state = check_random_state(random_state)
    int_part = int(np.floor(x))
    frac_part = x - int_part
    return int_part + int(random_state.random() < frac_part)

'''
Perform stratified sampling on dataframe df with respect to the variable in stratify_by.
This can also guarantee the presence of the target_var if it is binary and specified.
'''
def stratified_sample(df, n, stratify_by='grp_name', replace=False, train_test=False, random_state=None):
    while True:
        state = check_random_state(random_state)
        new_df = df.groupby(by=stratify_by, group_keys=False).apply(
            lambda x: x.sample(n=rnd_round(x.shape[0]/df.shape[0] * n, random_state=state),
                               replace=replace,
                               random_state=state))
        if train_test and not replace:
            test_index = df.index.difference(new_df.index)
            test_df = df.loc[test_index,:]
            if len(test_df):
                return new_df, test_df
        else:
            return new_df
        random_state += FOLDS


def split_df(df, metric, feature_names=[]):
    split_df = df.groupby('grp_name')
    X = split_df[feature_names].mean()
    counts = split_df['grp_name'].count()
    if metric is None:
        z = None
    elif type(metric) == str:
        z = split_df[metric].mean()
    else:
        z = split_df.apply(lambda x: metric(x.y_true, x.y_pred))
        select_na = z.isna()
        z[select_na] = 0.0
        counts[select_na] = 0
    return X, z, counts

def bootstrap_variance(boot_mu, counts, uniform_var, invalid_val=np.nan):
    vars = boot_mu.var(axis=1)
    valid_vars = (counts>1) & (~vars.isna())
    vars[~valid_vars] = 0.0
    
    sigma2_single = (vars*counts.pow(2))[valid_vars].sum() / (counts-1)[valid_vars].sum()

    if uniform_var:
        sigma2 = sigma2_single / counts
        sigma2_inv = counts / sigma2_single
        valid_sigma = (counts>0)
    else:
        sigma2 = vars*counts/(counts-1)
        sigma2_inv = 1 / sigma2
        valid_sigma = valid_vars & (vars>EPS)
    
    sigma2[~valid_sigma] = invalid_val
    sigma2_inv[~valid_sigma] = invalid_val
    return sigma2, sigma2_inv, valid_sigma

def structured_regression(mts, alphas=[20.0, 10.0, 5.0, 2.0, 1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005], order=float('inf'), weighted=True):

    output = mts.pooled_estimator
    features = [key for key in mts.dfs[0].keys()[4:] if key.count('None') >= len(mts.groups[0])-order or key.count('None') == 0]

    for t in range(mts.T):
        df = mts.dfs[t]
        n = df.shape[0]
        X, z, counts = split_df(df=df, metric=mts.metric, feature_names=features)
        index = z.index

        select = (counts>0)
        naive_boot = pd.DataFrame(index=index)
        for b in range(100):
            df_b = stratified_sample(df, n, replace=True, random_state=b)
            _, mu_b, _ = split_df(df=df_b, metric=mts.metric)
            naive_boot[b] = mu_b
        var, _, _ = bootstrap_variance(boot_mu=naive_boot, counts=counts, uniform_var=True)
        
        if weighted:
            sample_weight = counts
        else:
            sample_weight = 1.0*(counts > 0)
        sample_weight = sample_weight / sample_weight.mean()
        
        # ElasticNet with sample_weight = sw minimizes the following problem:
        #   1/2 * [sum_i sw_i * (y_i - x_i @ beta)^2] / [sum_i sw_i] + alpha * penalty
        # where
        #   penalty = l1_ratio * ||beta||_1 + 1/2 * (1 - l1_ratio) * ||beta||^2_2
        #
        # The expected value of the first term is:
        #   1/2 * [sum_i sw_i * var_i] / [sum_i sw_i]
        # So, equalizing the loss and penalty, for ridge (when l1_ratio = 0), we should approximately have
        #   alpha = [sum_i sw_i * var_i] / [sum_i sw_i] / ||beta_OLS||^2_2
        #
        # The lasso with
        #   1/2 * [sum_i sw_i * (y_i - x_i @ beta)^2] + lambda * penalty
        # Should have approximately
        #   lambda = max_j sqrt[sum_i sw_i^2 * var_i * x_ij^2]
        # So, adjusting for rescaling:
        #   alpha = (max_j ...) / [sum_i sw_i]
        
        X_col_norm = X.pow(2).mul(sample_weight.pow(2) * var, axis=0).loc[select,:].sum().pow(0.5)
        alpha_mul = X_col_norm.max() / sample_weight[select].sum()
        cv_res = pd.DataFrame(index=range(FOLDS), columns=alphas)
        for i in range(FOLDS):
            train, test = stratified_sample(df, FRAC*n, train_test=True, random_state=i)
            X_train, z_train, counts_train = split_df(df=train, metric=mts.metric, feature_names=features)
            X_test, z_test, counts_test = split_df(df=test, metric=mts.metric, feature_names=features)
            sample_weight_train = sample_weight[z_train.index] * (counts_train>0).astype(float) 
            sample_weight_test = sample_weight[z_test.index] * (counts_test>0).astype(float)
            for alpha in alphas:
                with warnings.catch_warnings():
                    warnings.simplefilter(action='ignore', category=ConvergenceWarning)
                    lasso = ElasticNet(alpha=alpha_mul*alpha, **LASSO_KWARGS).fit(X_train, z_train, sample_weight=sample_weight_train)
                z_predict = pd.Series(lasso.predict(X_test), index=X_test.index)
                cv_res.loc[i,alpha] = sample_weight_test @ (z_test-z_predict).pow(2)
        cv_means = cv_res.mean().astype(float)
        alpha_cv = cv_means.idxmin()

        with warnings.catch_warnings():
            warnings.simplefilter(action='ignore', category=ConvergenceWarning)
            lasso = ElasticNet(alpha=alpha_mul*alpha_cv, **LASSO_KWARGS).fit(X, z, sample_weight=sample_weight)
        output[t,mts.counts[t]>0] = lasso.predict(X)

    return output
