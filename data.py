import csv
import os
import pdb
import pickle
import sys
from argparse import ArgumentParser
from collections import namedtuple
from functools import cache, cached_property
from itertools import chain, combinations, product
import jiwer
import numpy as np; np.seterr(all='raise')
import pandas as pd
from fairlearn import datasets as fld
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegressionCV, RidgeCV
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler


MIN_COUNT = 40
MIN_LABEL = 4

MODELS = {'logit': Pipeline([('scaler', StandardScaler()),
                             ('model', LogisticRegressionCV())]),
          'ridge': Pipeline([('scaler', StandardScaler()),
                             ('model', RidgeCV())])}
STATES = {"AL": "01", "AK": "02", "AZ": "04", "AR": "05", "CA": "06", "CO": "08", "CT": "09", "DE": "10", "FL": "12", "GA": "13", "HI": "15", "ID": "16", "IL": "17", "IN": "18", "IA": "19", "KS": "20", "KY": "21",
          "LA": "22", "ME": "23", "MD": "24", "MA": "25", "MI": "26", "MN": "27", "MS": "28", "MO": "29", "MT": "30", "NE": "31", "NV": "32", "NH": "33", "NJ": "34", "NM": "35", "NY": "36", "NC": "37", "ND": "38", 
          "OH": "39", "OK": "40", "OR": "41", "PA": "42", "RI": "44", "SC": "45", "SD": "46", "TN": "47", "TX": "48", "UT": "49", "VT": "50", "VA": "51", "WA": "53", "WV": "54", "WI": "55", "WY": "56", "PR": "72"}

RACEMAP = {'africanamerican': 1,
           'asian': 2,
           'caucasian': 3,
           'hispanic': 4,
           'other': 5,
           'unknown': 6,
           ' Amer-Indian-Eskimo': 1, 
           ' Asian-Pac-Islander': 2, 
           ' Black': 3, 
           ' Other': 4,
           ' White': 5} 
SEXMAP = {'': 0,
          'female': 1,
          'male': 2,
          'other': 3,
          'unknown/invalid': 3,
          ' Female': 1,
          ' Male': 2}
AGEMAP = {'': 0,
          'teens': 1,
          'twenties': 2,
          'thirties': 3,
          'fourties': 4,
          'fifties': 5,
          'sixties': 6,
          'seventies': 7,
          'eighties': 8,
          'nineties': 9,
          '30 years or younger': 1,
          '30-60 years': 2,
          'over 60 years': 3}


def mpi4py_reduce(array, comm, all=False):
    if not comm is None and comm.size > 1:
        from mpi4py import MPI
        recvbuf = None if comm.rank and not all else [np.empty(array.shape), MPI.DOUBLE]
        (comm.Allreduce if all else comm.Reduce)([array, MPI.DOUBLE], recvbuf)
        return None if comm.rank and not all else recvbuf[0]
    return array


def auc_metric(labels, scores):
    if len(np.unique(labels)) == 1:
        return np.nan
    return roc_auc_score(labels, scores)

def error_metric(labels, predictions):
    return np.absolute(predictions - labels).mean()


class MultiTaskStatistics:
    def __init__(self, errors, labels=None, scores=None, groups=None, statistic='error'):
        self.errors = errors
        self.T = len(errors)
        self.d = len(errors[0])
        self.labels = labels
        self.scores = scores
        self.groups = [(g,) for g in range(len(errors[0]))] if groups is None else groups
        self._statistic = statistic
        self.bounds = (0., float('inf')) if statistic == 'error' else (0., 1.)
        self.theta = np.zeros(self.d) + (statistic == 'auc')
        self.metric = error_metric if statistic == 'error' else auc_metric

    @cached_property
    def binary(self):
        return all(error in {0, 1} 
                   for errors in self.errors
                   for group_errors in errors 
                   for error in group_errors)

    @cached_property
    def ones(self):
        return np.ones((self.T, self.d))

    @cached_property
    def eye(self):
        return np.stack([np.eye(self.d)] * self.T)

    @cached_property
    def counts(self):
        output = np.empty((self.T, self.d))
        for t, errors in enumerate(self.errors):
            output[t] = np.array([len(group_errors) for group_errors in errors])
        return output

    @cache
    def label_counts(self, label):
        output = np.empty((self.T, self.d))
        for t, labels in enumerate(self.labels):
            output[t] = np.array([(group_labels == label).sum()
                                  for group_labels in labels])
        return output

    @cached_property
    def counts01(self):
        return self.label_counts(0) * self.label_counts(1)

    @cached_property
    def means(self):
        output = np.empty((self.T, self.d))
        for t, (counts, errors) in enumerate(zip(self.counts, self.errors)):
            output[t] = np.array([group_errors.mean() if gt0 else float('nan')
                                  for gt0, group_errors in zip(counts>0, errors)])
        return output

    @cached_property
    def aucs(self):
        output = np.empty((self.T, self.d))
        for t, (labels, scores, counts01) in enumerate(zip(self.labels, self.scores, self.counts01)):
            output[t] = np.array([roc_auc_score(group_labels, group_scores) if gt0 else float('nan')
                                  for gt0, group_labels, group_scores in zip(counts01 > 0, labels, scores)])
        return output

    @cached_property
    def auc(self):
        output = np.empty((self.T, self.d))
        for t, (labels, scores) in enumerate(zip(self.labels, self.scores)):
            output[t] = roc_auc_score(np.concatenate(labels), np.concatenate(scores))
        return output

    @cached_property
    def variances(self):
        output = np.empty((self.T, self.d))
        if self._statistic == 'error':
            for t, (counts, errors) in enumerate(zip(self.counts, self.errors)):
                gt1s = counts > 1
                group_variances = np.array([group_errors.std() ** 2 
                                            for gt1, group_errors in zip(gt1s, errors) if gt1])
                if group_variances.any():
                    shared_variance = (group_variances * (counts[gt1s]-1)).sum() / (counts[gt1s]-1).sum()
                else:
                    total = counts.sum()
                    shared_variance = np.std(np.concatenate(errors)) ** 2 * total / (total-1)
                gt0s = counts > 0
                output[t, gt0s] = shared_variance / counts[gt0s]
                output[t, ~gt0s] = float('nan')
        elif self._statistic == 'auc':
            for t, (counts, counts01) in enumerate(zip(self.counts, self.counts01)):
                gt0s = counts01 > 0
                output[t, gt0s] = (counts[gt0s]+1.) / counts01[gt0s] / 12. 
                output[t, ~gt0s] = float('nan')
        else:
            raise(NotImplementedError)
        return output 

    @cached_property
    def nan(self):
        if self._statistic == 'error':
            return self.counts == 0
        if self._statistic == 'auc':
            return self.counts01 == 0
        raise(NotImplementedError)

    @cached_property
    def notnan(self):
        return ~self.nan

    @cached_property
    def means_zero_nans(self):
        output = np.copy(self.means)
        output[~self.notnan] = 0.
        return output

    @cached_property
    def aucs_zero_nans(self):
        output = np.copy(self.aucs)
        output[~self.notnan] = 0.
        return output

    @cached_property
    def y(self):
        if self._statistic == 'error':
            return self.means_zero_nans
        if self._statistic == 'auc':
            return self.aucs_zero_nans
        raise(NotImplementedError)

    @cached_property
    def variances_zero_nans(self):
        output = np.copy(self.variances)
        output[~self.notnan] = 0.
        return output

    @cached_property
    def shared_variance(self):
        return (self.variances_zero_nans * self.unscaled_precision).max(1)

    @cached_property
    def unscaled_precision(self):
        if self._statistic == 'error':
            return self.counts
        if self._statistic == 'auc':
            return self.counts01 / (self.counts+1.)
        raise(NotImplementedError)

    @cached_property
    def pooled_estimator(self):
        return ((self.unscaled_precision * self.y).sum(1) / self.unscaled_precision.sum(1))[:,None] * self.ones

    @cached_property
    def deff(self):
        return self.notnan.sum(1)

    @cached_property
    def inverse_variances(self):
        output = np.zeros((self.T, self.d))
        output[self.notnan] = 1. / self.variances[self.notnan]
        return output

    @cached_property
    def normalized_counts(self):
        return self.counts / self.counts.sum(1)[:,None]

    @cached_property
    def normalized_precision(self):
        return self.unscaled_precision / self.unscaled_precision.sum(1)[:,None]

    @cached_property
    def indicators(self):
        groups = np.array([[label for label in group if not label is None]
                           for group in self.groups])
        num_groups, num_attrs = groups.shape
        output = []
        for k in range(0, num_attrs+1):
            intersections = []
            for attrs in combinations(range(num_attrs), k):
                indicators = []
                for labels in product(*chain(np.unique(groups[:,attr])
                                             for attr in attrs)):
                    indicator = np.ones(num_groups)
                    for attr, label in zip(attrs, labels):
                        indicator *= groups[:,attr] == label
                    indicators.append(indicator)
                intersections.append(np.array(indicators))
            output.append(intersections)
        return output

    @cached_property
    def num_attrs(self):
        return len(self.indicators)-1

    @cache
    def UTs_idx(self, order=float('inf')):
        if order == -1:
            return np.empty((0, self.d)), np.empty(0)
        UTs, idx = zip(*((indic, len(indic)) for indics in self.indicators[:min(order+1, self.num_attrs)]
                                             for indic in indics))
        return np.vstack(UTs), np.cumsum(idx)

    @cached_property
    def dfs(self):

        groups = [tuple(a for a in group if not a is None)
                  for group in self.groups]
        classes = [sorted({group[i] for group in groups})
                   for i in range(self.num_attrs)]
        features = sum((list(product(*[classes[i] if i in combination else [None]
                                       for i in range(self.num_attrs)]))
                        for j in range(self.num_attrs+1)
                        for combination in combinations(range(self.num_attrs), j)), [])
        fnames = [' '.join(str(attr) for attr in feature)
                  for feature in features]

        output = []
        for t, errors in enumerate(self.scores if self._statistic == 'auc' else self.errors):
            datadict = {'grp_name': [], 'y_true': [], 'y_pred': []}
            for fname in fnames:
                datadict[fname] = []
            for i, (group, group_errors) in enumerate(zip(groups, errors)):
                n = len(group_errors)
                if n:
                    for fname, feature in zip(fnames, features):
                        entry = all(fattr is None or gattr == fattr
                                    for gattr, fattr in zip(group, feature))
                        datadict[fname] += [entry for _ in range(n)]
                    grp_name = ' '.join(str(attr) for attr in group)
                    datadict['grp_name'] += [grp_name for _ in range(n)]
                    datadict['y_true'] += list(self.labels[t][i]) if self._statistic == 'auc' else [0. for _ in range(n)]
                    for group_error in group_errors:
                        datadict['y_pred'].append(group_error)
            output.append(pd.DataFrame(data=datadict))

        return output


def randomized_round(f):
    return int(f) + (np.random.rand() < f-int(f))


def subsample(xs, rate):
    data = np.concatenate(xs)
    groups = np.concatenate([i * np.ones(len(x), int) 
                             for i, x in enumerate(xs)])
    xs = [[] for _ in range(len(xs))]
    for i in np.random.choice(len(data), size=randomized_round(rate*len(data))):
        xs[groups[i]].append(data[i])
    return [np.array(x, dtype=data.dtype) for x in xs]


class Dataset:

    group_types = ['all', 'large', 'small']
    _statistic = 'error'

    @cached_property
    def groups(self):
        return sorted(self.group_errors(self.tasks[0]).keys())

    @cached_property
    def num_groups(self):
        return len(self.groups)

    @cache
    def errors(self, task):
        return [self.group_errors(task)[group] for group in self.groups]

    @cache
    def group_counts(self, task):
        return np.array([len(errors) for errors in self.errors(task)])

    @cache
    def large_enough_groups(self, task):
        output = self.group_counts(task) >= MIN_COUNT
        if self._statistic == 'auc':
            return np.logical_and(output,
                                  [min(sum(labels == 0), sum(labels == 1)) >= MIN_LABEL
                                   for labels in self.labels(task)])
        return output

    @cache
    def small_groups(self, task):
        counts = self.group_counts(task)
        enough = self.large_enough_groups(task)
        return np.logical_and(enough, counts <= np.median(counts[enough]))

    @cache
    def large_groups(self, task):
        counts = self.group_counts(task)
        enough = self.large_enough_groups(task)
        return np.logical_and(enough, counts > np.median(counts[enough]))

    @cache
    def group_mask(self, groups):
        get_mask = getattr(self, groups+'_groups')
        return np.array([get_mask(task) for task in self.tasks])

    @cache
    def group_labels(self, task):
        return {group: np.logical_xor(self.group_errors(task)[group],
                                      self.group_scores(task)[group] > .5)
                for group in self.groups}

    @cache
    def scores(self, task):
        return [self.group_scores(task)[group] for group in self.groups]

    @cache
    def labels(self, task):
        return [self.group_labels(task)[group] for group in self.groups]

    @cache
    def ground_truth(self, task):
        if self._statistic == 'error':
            return np.array([errors.mean() if len(errors) >= MIN_COUNT else float('nan')
                             for errors in self.errors(task)])
        elif self._statistic == 'auc':
            return np.array([roc_auc_score(labels, scores) if len(labels) >= MIN_COUNT and min(sum(labels == 0), sum(labels == 1)) >= MIN_LABEL else float('nan')
                             for labels, scores in zip(self.labels(task), self.scores(task))])
        else:
            raise(NotImplementedError)

    @cached_property
    def ground_truths(self):
        return np.array([self.ground_truth(task) for task in self.tasks])

    def mses(self, estimates, groups='all', tasks=None):
        if tasks is None:
            tasks = self.tasks
        tasks = set(tasks)
        task_mask = np.array([task in tasks for task in self.tasks])
        differences = self.ground_truths[task_mask] - estimates
        group_mask = (self.group_mask('large_enough' if groups == 'all' else groups))[task_mask]
        differences[~group_mask] = 0.
        return (differences*differences).sum(1) / group_mask.sum(1)

    def maes(self, estimates, groups='all', tasks=None):
        if tasks is None:
            tasks = self.tasks
        tasks = set(tasks)
        task_mask = np.array([task in tasks for task in self.tasks])
        differences = self.ground_truths[task_mask] - estimates
        group_mask = (self.group_mask('large_enough' if groups == 'all' else groups))[task_mask]
        differences[~group_mask] = 0.
        return np.absolute(differences).sum(1) / group_mask.sum(1)

    def subsample(self, rate, tasks=None):
        lists = [[]] + [[] if self._statistic == 'auc' else None for _ in range(2)]
        funcs = [self.errors, self.labels, self.scores]
        for task in (self.tasks if tasks is None else tasks):
            idx = subsample([np.arange(count) for count in self.group_counts(task)], rate)
            for data, func in zip(lists, funcs):
                if type(data) == list:
                    data.append([group_data[group_idx] for group_data, group_idx in zip(func(task), idx)])
        return MultiTaskStatistics(lists[0], labels=lists[1], scores=lists[2], groups=self.groups, statistic=self._statistic)


def write(string, rank=0):
    if not rank:
        sys.stdout.write(string)
        sys.stdout.flush()


def remove_tail_repetitions(string):

    for i in range(1, 1+len(string)//2):
        if string[-i:] == string[-2*i:-i]:
            sub = string[-i:]
            break
    else:
        return string

    start = min(string.index(sub[i:]+sub[:i]) 
                for i in range(len(sub)))
    return string[:start+len(sub)]


class WhisperCV(Dataset):

    def __init__(self,
                 disaggregate_sex=True,
                 disaggregate_age=True,
                 cache_dir='./cache/WhisperCV',
                 cer=False,
                 verbose=True,
                 ):

        self._cache = cache_dir
        self._sex = disaggregate_sex
        self._age = disaggregate_age
        self._verbose = verbose
        os.makedirs(cache_dir, exist_ok=True)
        self.name = 'Common Voice ASR (WER)'
        self._cer = cer
        self._metric = jiwer.cer if cer else jiwer.wer
        if cer:
            self.name = self.name.replace('WER', 'CER')

    @cached_property
    def groups(self):
        sexes = range(1, 4) if self._sex else [None]
        ages = range(1, 10) if self._age else [None]
        return list(product(sexes, ages))

    @cache
    def group_errors(self, *args):
        '''computes group errors
        args:
            no effect; present for convenience
        output:
            dict of group key: numpy array of errors; keys are int doubles 
            indexing (sex, age), with None if not split by that category
        '''

        fname = os.path.join(self._cache, 
                             f'{"cer-" if self._cer else ""}{int(self._sex)}{int(self._age)}.pkl')
        if os.path.isfile(fname):
            with open(fname, 'rb') as f:
                output = pickle.load(f)
        else:
            data = {}
            client_ids = {}
            sents, preds, ages, sexes = [], [], [], []
            for subset in ['test', 'validation']:
                with open(f'data/common_voice/whisper-en-{subset}.pkl', 'rb') as f:
                    for key, value in pickle.load(f).items():
                        data[key] = np.concatenate([data.get(key, []), value])
            clients = {client_id: i for i, client_id in enumerate(sorted(set(data['client_ids'])))}
            sentslist = [[] for _ in clients]
            predslist = [[] for _ in clients]
            array = np.zeros((len(clients), 3))
            for client_id, age, sex, sent, pred in zip(data['client_ids'],
                                                       data['ages'],
                                                       data['genders'],
                                                       data['sentences'],
                                                       data['predictions']):
                i = clients[client_id]
                if age:
                    array[i,1] = AGEMAP[age]
                if sex:
                    array[i,2] = SEXMAP[sex]
                sentslist[i].append(sent)
                predslist[i].append(remove_tail_repetitions(pred))
            for i, (sents, preds) in enumerate(zip(sentslist, predslist)):
                array[i,0] = self._metric(sents, preds) if sents and any(array[i,1:]) else float('nan')
            array = array[~np.isnan(array[:,0])]
            output = {}
            for key in self.groups:
                sex, age = key
                select = np.ones(len(array), dtype=bool)
                select = select if sex is None else np.logical_and(select, array[:,2] == sex)
                select = select if age is None else np.logical_and(select, array[:,1] == age)
                output[key] = array[select,0]
            with open(fname, 'wb') as f:
                pickle.dump(output, f)
        return output

    @cached_property
    def tasks(self):
        return np.array([0])


class WhisperCVC(Dataset):

    def __init__(self,
                 disaggregate_sex=True,
                 disaggregate_age=True,
                 cache_dir='./cache/WhisperCVC',
                 clusters=20,
                 alpha=.5,
                 cer=False,
                 verbose=True,
                 ):

        self._cache = cache_dir
        self._clusters = clusters
        self._alpha = alpha
        self._sex = disaggregate_sex
        self._age = disaggregate_age
        self._verbose = verbose
        os.makedirs(cache_dir, exist_ok=True)
        self.name = 'Common Voice Clusters ASR (WER)'
        self._cer = cer
        self._metric = jiwer.cer if cer else jiwer.wer
        if cer:
            self.name = self.name.replace('WER', 'CER')

    def embeddings(self, data):
        fname = os.path.join(self._cache, 'embeddings.npy')
        if os.path.isfile(fname):
            with open(fname, 'rb') as f:
                embeddings = np.load(f)
        else:
            model = SentenceTransformer('average_word_embeddings_glove.840B.300d')
            embeddings = model.encode(data['sentences'])
            with open(fname, 'wb') as f:
                np.save(f, embeddings)
            if self._verbose:
                write('computed embeddings\n')
        return embeddings

    def clusters(self, data):
        fname = os.path.join(self._cache,
                             f'{self._clusters}_clusters.npy')
        if os.path.isfile(fname):
            with open(fname, 'rb') as f:
                clusters = np.load(f)
        else:
            embeddings = self.embeddings(data)
            kmeans = KMeans(n_clusters=self._clusters, max_iter=100, n_init='auto')
            clusters = kmeans.fit_predict(embeddings)
            with open(fname, 'wb') as f:
                np.save(f, clusters)
            if self._verbose:
                write('clustered sentences\n')
        return clusters

    @cached_property
    def client_array(self):
        fname = os.path.join(self._cache, 
                             f'{self._clusters}-{self._alpha}_{"cer" if self._cer else "wer"}.npy')
        if os.path.isfile(fname):
            with open(fname, 'rb') as f:
                array = np.load(f)
        else:
            with open('data/common_voice/whisper-en-train.pkl', 'rb') as f:
                data = pickle.load(f)
            clusters = self.clusters(data)
            clients = {pair: i for i, pair in enumerate(product(set(data['client_ids']), range(self._clusters)))}
            sentslist = [[] for _ in clients]
            predslist = [[] for _ in clients]
            array = np.zeros((len(clients), 4))
            ds = (1.-self._alpha)*np.eye(self._clusters) + self._alpha/self._clusters
            assign = lambda cluster: np.random.choice(self._clusters, p=ds[:,cluster])
            for client_id, age, sex, sent, pred, cluster in zip(data['client_ids'],
                                                                data['ages'],
                                                                data['genders'],
                                                                data['sentences'],
                                                                data['predictions'],
                                                                clusters):
                cluster = assign(cluster)
                i = clients[(client_id, cluster)]
                sentslist[i].append(sent)
                predslist[i].append(remove_tail_repetitions(pred))
                if age:
                    array[i,1] = AGEMAP[age]
                if sex:
                    array[i,2] = SEXMAP[sex]
                array[i,3] = cluster
            for i, (sents, preds) in enumerate(zip(sentslist, predslist)):
                array[i,0] = self._metric(sents, preds) if sents and any(array[i,1:]) else float('nan')
            array = array[~np.isnan(array[:,0])]
            with open(fname, 'wb') as f:
                np.save(f, array)
            if self._verbose:
                write('processed predictions\n')
        return array

    @cached_property
    def groups(self):
        sexes = range(1, 4) if self._sex else [None]
        ages = range(1, 10) if self._age else [None]
        return list(product(sexes, ages))

    @cache
    def group_errors(self, task):
        fname = os.path.join(self._cache, 
                             f'{"cer-" if self._cer else ""}{self._clusters}-{self._alpha}_{task}{int(self._age)}{int(self._sex)}.pkl')
        if os.path.isfile(fname):
            with open(fname, 'rb') as f:
                output = pickle.load(f)
        else:
            output = {}
            errors = self.client_array[self.client_array[:,3] == task]
            for group in self.groups:
                sex, age = group 
                select = np.ones(len(errors), dtype=bool)
                select = select if sex is None else np.logical_and(select, errors[:,2] == sex)
                select = select if age is None else np.logical_and(select, errors[:,1] == age)
                output[group] = errors[select,0]
            with open(fname, 'wb') as f:
                pickle.dump(output, f)
        return output

    @cached_property
    def tasks(self):
        return np.arange(self._clusters)


class Diabetes(Dataset):

    def __init__(self,
                 disaggregate_race=True,
                 disaggregate_sex=True,
                 disaggregate_age=True,
                 cache_dir='./cache/Diabetes',
                 model='logit',
                 mse=False,
                 auc=False
                 ):
        '''
            disaggregate_race: whether to split groups by race
            disaggregate_sex: whether to split groups by sex
            disaggregate_age: whether to split groups by age
        '''

        self._cache = cache_dir
        self._model = model
        self._mse = mse
        self._statistic = 'auc' if auc else 'error'
        self._race = disaggregate_race
        self._sex = disaggregate_sex
        self._age = disaggregate_age
        os.makedirs(cache_dir, exist_ok=True)
        self.name = f'Diabetes {"Classification (0-1 error)" if model == "logit" else "Regression"}'
        if "Regression" in self.name:
            self.name += f' ({"MSE" if mse else "MAE"})'
        if auc:
            self.name = self.name.replace('0-1 error', 'AUC')

    @cache
    def group_errors(self, *args):
        '''computes group errors
        args:
            no effect; present for convenience
        output:
            dict of group key: numpy array of errors; keys are int triples
            indexing (race, sex, age), with None if not split by that category
        '''

        fname = os.path.join(self._cache, 
                             f'{self._model}{"_MSE" if self._mse else ""}-{int(self._race)}{int(self._sex)}{int(self._age)}.pkl')

        if os.path.isfile(fname):
            with open(fname, 'rb') as f:
                output = pickle.load(f)

        else:
            grpdata = []
            numdata = []
            catdata = []
            with open('data/diabetic_preprocessed.csv', 'r') as f:
                reader = csv.reader(f)
                header = next(reader)[3:]
                for row in csv.reader(f):
                    grpdata.append([' '.join(row[:3])])
                    numdata.append([row[5]] + row[7:10] + [row[11]])
                    catdata.append(row[3:5] + [row[6], row[10]] + row[12:])
            ordinal = OrdinalEncoder()
            onehot = OneHotEncoder(drop='if_binary', sparse_output=False) 
            if self._model == 'ridge':
                data = np.hstack([ordinal.fit_transform(grpdata),
                                  np.array(numdata).astype(np.float64),
                                  onehot.fit_transform(catdata)])
            elif self._model == 'logit':
                data = np.hstack([ordinal.fit_transform(grpdata),
                                  onehot.fit_transform(catdata),
                                  np.array(numdata).astype(np.float64)])
            else:
                raise(NotImplementedError)
            X, y = data[:,2:], data[:,1]
            model = MODELS[self._model].fit(X, y)
            errs = np.absolute(model.predict(X) - y)
            if self._mse:
                errs *= errs
            output = {}
            clf = hasattr(model, 'predict_proba')
            if clf:
                proba = model.predict_proba(X)[:,1]
                scores = {}
            for i, group in enumerate(ordinal.categories_[0]):
                split = group.lower().split()
                race = RACEMAP[split[0]] if self._race else None
                sex = SEXMAP[split[1]] if self._sex else None
                age = AGEMAP[' '.join(split[2:])] if self._age else None
                key = (race, sex, age)
                output[key] = np.concatenate([output.get(key, []), errs[data[:,0] == i]])
                if clf:
                    scores[key] = np.concatenate([scores.get(key, []), proba[data[:,0] == i]])
            if clf:
                with open(fname.replace('-', '_scores-'), 'wb') as f:
                    pickle.dump(scores, f)
            with open(fname, 'wb') as f:
                pickle.dump(output, f)

        return output

    @cache
    def group_scores(self, *args):
        self.group_errors(*args)
        with open(os.path.join(self._cache, f'{self._model}_scores-{int(self._race)}{int(self._sex)}{int(self._age)}.pkl'), 'rb') as f:
            output = pickle.load(f)
        return output

    @cached_property
    def tasks(self):
        return np.array([0])


class AdultLlama(Dataset):

    def __init__(self,
                 disaggregate_race=True,
                 disaggregate_sex=True,
                 disaggregate_age=True,
                 cache_dir='./cache/AdultLlama',
                 model='logit',
                 mse=False,
                 ):
        '''
            disaggregate_race: whether to split groups by race
            disaggregate_sex: whether to split groups by sex
            disaggregate_age: whether to split groups by age
        '''

        self._cache = cache_dir
        self._model = model
        self._mse = mse
        self._race = disaggregate_race
        self._sex = disaggregate_sex
        self._age = disaggregate_age
        os.makedirs(cache_dir, exist_ok=True)
        self.name = f'Adult Classification (0-1 error)'

    @cache
    def group_errors(self, *args):
        '''computes group errors
        args:
            no effect; present for convenience
        output:
            dict of group key: numpy array of errors; keys are int triples
            indexing (race, sex, age), with None if not split by that category
        '''

        fname = os.path.join(self._cache,
                             f'{int(self._race)}{int(self._sex)}{int(self._age)}.pkl')
        if os.path.isfile(fname):
            with open(fname, 'rb') as f:
                output = pickle.load(f)
        else:
            df = pd.read_csv('data/adult_llama.csv')
            races = sorted(set(df['race'])) if self._race else [None]
            sexes = sorted(set(df['sex'])) if self._sex else [None]
            ages = np.array([1, 2, 3]) if self._age else [None]
            errs = np.array(df['label'] != df['pred'])
            output = {}
            for i, group in enumerate(product(races, sexes, ages)):
                race, sex, age = group
                select = np.ones(len(df), dtype=bool)
                select = select if race is None else np.logical_and(select, df['race'] == race)
                select = select if sex is None else np.logical_and(select, df['sex'] == sex)
                if age == 1:
                    select = np.logical_and(select, df['age'] < 25)
                elif age == 2:
                    select = np.logical_and(select, np.logical_and(df['age'] >= 25, df['age'] < 65))
                elif age == 3:
                    select = np.logical_and(select, df['age'] >= 65)
                output[group] = errs[select]
            with open(fname, 'wb') as f:
                pickle.dump(output, f)
        return output

    @cached_property
    def tasks(self):
        return np.array([0])


class StateLevelACS(Dataset):

    _cats = {
             "MAR": np.arange(1, 6),
             "RELP": np.arange(18),
             "RAC1P": np.arange(1, 10),
             "SEX": np.arange(1, 3),
             }

    def __init__(self, 
                 disaggregate_race=True,
                 disaggregate_sex=True,
                 disaggregate_age=True,
                 cache_dir='./cache/SLACS', 
                 income_threshold=50000, 
                 state='CA', 
                 model='logit',
                 auc=False,
                 ):
        '''
            disaggregate_race: whether to split groups by race
            disaggregate_sex: whether to split groups by sex
            disaggregate_age: whether to split groups by age
        '''

        self._cache = cache_dir
        self._threshold = income_threshold
        self._state = state
        self._model = model
        self._statistic = 'auc' if auc else 'error'
        self._race = disaggregate_race
        self._sex = disaggregate_sex
        self._age = disaggregate_age
        os.makedirs(cache_dir, exist_ok=True)
        self.name = 'State-Level ACS Classification (0-1 error)'
        if auc:
            self.name = self.name.replace('0-1 error', 'AUC')

    @cache
    def raw(self, state):
        return fld.fetch_acs_income(states=[state])

    @cache
    def X(self, state):
        fname = os.path.join(self._cache, 
                             f'{state}_X.pkl')
        if os.path.isfile(fname):
            X = pd.read_pickle(fname)
        else:
            raw = self.raw(state)
            df = raw.data.drop(columns=['COW', 'SCHL', 'POBP', 'OCCP']).astype(int)
            df['over_25'] = df['AGEP'] >= 25
            for col, cats in self._cats.items():
                df[col] = df[col].astype(pd.api.types.CategoricalDtype(cats))
            X = pd.get_dummies(df)
            X.to_pickle(fname)
        return X

    @cache
    def y(self, state):
        fname = os.path.join(self._cache, 
                             f'{state}-{self._threshold}_y.pkl')
        if os.path.isfile(fname):
            y = pd.read_pickle(fname)
        else:
            y = (self.raw(state).target > self._threshold).astype(int)
            y.to_pickle(fname)
        return y

    @cached_property
    def model(self):
        fname = os.path.join(self._cache, 
                             f'{self._model}-{self._state}-{self._threshold}.pkl')
        if os.path.isfile(fname):
            with open(fname, 'rb') as f:
                model = pickle.load(f)
        else:
            X, y = self.X(self._state), self.y(self._state)
            model = MODELS[self._model].fit(X, y)
            with open(fname, 'wb') as f:
                pickle.dump(model, f)
        return model

    @cached_property
    def groups(self):
        races = self._cats['RAC1P'] if self._race else [None]
        sexes = self._cats['SEX'] if self._sex else [None]
        ages = np.array([1, 2, 3]) if self._age else [None]
        return list(product(races, sexes, ages))

    @cache
    def group_errors(self, state):
        '''computes group errors of a given state
        args:
            state: two-letter state code
        output:
            dict of group key: numpy array of errors; keys are int triples 
            indexing (race, sex, age), with None if not split by that category
        '''

        suffix = f'{state}{int(self._race)}{int(self._sex)}{int(self._age)}'
        fname = os.path.join(self._cache,
                             f'{self._model}-{self._state}-{self._threshold}_{suffix}.pkl')
        if os.path.isfile(fname):
            with open(fname, 'rb') as f:
                output = pickle.load(f)
        else:
            X = self.X(state)
            errs = (self.model.predict(X) != self.y(state)).astype(int)
            output = {}
            proba = self.model.predict_proba(X)[:,1]
            scores = {}
            for group in self.groups:
                race, sex, age = group 
                select = np.ones(len(errs), dtype=bool)
                select = select if race is None else np.logical_and(select, X[f'RAC1P_{race}'])
                select = select if sex is None else np.logical_and(select, X[f'SEX_{sex}'])
                if age == 1:
                    select = np.logical_and(select, X['AGEP'] < 25)
                elif age == 2:
                    select = np.logical_and(select, np.logical_and(X['AGEP'] >= 25, X['AGEP'] < 65))
                elif age == 3:
                    select = np.logical_and(select, X['AGEP'] >= 65)
                output[group] = errs[select].to_numpy()
                scores[group] = proba[select]
            with open(fname, 'wb') as f:
                pickle.dump(output, f)
            with open(fname.replace('.pkl', '_scores.pkl'), 'wb') as f:
                pickle.dump(scores, f)
        return output

    @cache
    def group_scores(self, state):
        self.group_errors(state)
        suffix = f'{state}{int(self._race)}{int(self._sex)}{int(self._age)}'
        fname = os.path.join(self._cache,
                             f'{self._model}-{self._state}-{self._threshold}_{suffix}_scores.pkl')
        with open(fname, 'rb') as f:
            output = pickle.load(f)
        return output

    @cached_property
    def tasks(self):
        return np.array([state for state in sorted(STATES.keys())
                         if state != self._state])


class AdditivePrior(Dataset):

    def __init__(self, d=20, T=100, tau_squared=[.125, 1., .5, .25], scale=1.):
        self.d = d
        self.T = T
        self.theta = 1. / (np.arange(d)+1.)
        self.tau_squared = tau_squared
        self.scale = scale
        self.Lambda = tau_squared[0] * np.eye(d) + tau_squared[1]
        U = np.zeros((self.d, self.d // 2))
        for i in range(self.d // 2):
            U[2*i,i] = 1.
            U[2*i+1,i] = 1.
        self.Lambda += tau_squared[2] * U.dot(U.T)
        U = np.zeros((self.d, 2))
        U[::2,0] = 1.
        U[1::2,1] = 1.
        self.Lambda += tau_squared[3] * U.dot(U.T)
        self._gt = np.random.multivariate_normal(self.theta, self.Lambda, self.T)
        self.ground_truths = self._gt
        self.data = [[np.random.normal(self.ground_truths[t, i], np.sqrt(scale), i+1) 
                      for i in range(d)] for t in range(T)]

    @cached_property
    def groups(self):
        return [(i, j) for i in range(self.d//2) for j in range(2)]

    @cache
    def group_errors(self, task):
        return {group: self.data[task][i]
                for i, group in enumerate(self.groups)}

    @cached_property
    def tasks(self):
        return np.arange(self.T)


def parse(parser=ArgumentParser(), verbose=True, default_dataset='SLACS'):
    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank, size = comm.rank, comm.size
    except ImportError:
        rank, size = 0, 1
        comm = namedtuple("no_mpi4py", ("rank", "size"))(rank, size)

    parser = ArgumentParser()
    parser.add_argument('--num-trials', type=int, default=None)
    parser.add_argument('--num-folds', type=int, default=5)
    parser.add_argument('--dataset', type=str, default=default_dataset)
    parser.add_argument('--disagg', type=str, default='rsa')
    parser.add_argument('--clip-wer', action='store_true')
    parser.add_argument('--min-log10-sample-rate', type=float, default=-2.)
    parser.add_argument('--max-log10-sample-rate', type=float, default=-0.)
    parser.add_argument('--num-sample-rates', type=int, default=9)
    parser.add_argument('--output-dir', type=str, default='/workspace/plots/')
    parser.add_argument('--no-plots', action='store_true')
    parser.add_argument('--sample-rate', type=float, default=.1)
    args = parser.parse_args()
    if args.num_trials is None:
        args.num_trials = size

    if args.dataset[:5] == 'SLACS':
        dataset = StateLevelACS(disaggregate_race='r' in args.disagg,
                                disaggregate_sex='s' in args.disagg,
                                disaggregate_age='a' in args.disagg,
                                auc=args.dataset[-4:] == '-AUC')
    elif args.dataset[:10] == 'WhisperCVC' and '-' in args.dataset:
        dataset = WhisperCVC(alpha=float(args.dataset.split('-')[1]),
                             disaggregate_sex='s' in args.disagg,
                             disaggregate_age='a' in args.disagg,
                             cer=args.dataset[-4:] == '-CER')
    elif args.dataset[:8] == 'Diabetes':
        dataset = Diabetes(disaggregate_race='r' in args.disagg,
                           disaggregate_sex='s' in args.disagg,
                           disaggregate_age='a' in args.disagg,
                           model='ridge' if args.dataset[8:18] == 'Regression' else 'logit',
                           mse=args.dataset[-4:] == '-MSE',
                           auc=args.dataset[-4:] == '-AUC')
    elif args.dataset == 'AdultLlama':
        dataset = AdultLlama(disaggregate_race='r' in args.disagg,
                             disaggregate_sex='s' in args.disagg,
                             disaggregate_age='a' in args.disagg)
    elif args.dataset[:9] == 'WhisperCV':
        dataset = WhisperCV(disaggregate_sex='s' in args.disagg,
                            disaggregate_age='a' in args.disagg,
                            cer=args.dataset[-4:] == '-CER')
    else:
        if verbose:
            write(f'dataset {args.dataset} not found\n', rank)
        dataset = None

    if not dataset is None:
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
        if verbose:
            write(f'finished loading {dataset.name} disaggregated by {args.disagg}\n', rank)

    args.sample_rates = 10. ** np.linspace(args.min_log10_sample_rate,
                                           args.max_log10_sample_rate,
                                           args.num_sample_rates)
    os.makedirs(args.output_dir, exist_ok=True)

    return args, dataset, comm
