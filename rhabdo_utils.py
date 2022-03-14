import numpy as np
import pandas as pd
import os
import tensorflow as tf
import seaborn as sns
import warnings
import scanpy as sc
from Bio import SeqIO
import math
from anndata import AnnData
from collections import Counter
import sys
import datetime
from dateutil.parser import parse as dparse
import errno
import matplotlib.pyplot as plt

def tprint(string):
    string = str(string)
    sys.stdout.write(str(datetime.datetime.now()) + ' | ')
    sys.stdout.write(string + '\n')
    sys.stdout.flush()

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def iterate_lengths(lengths, seq_len):
    curr_idx = 0
    for length in lengths:
        if length > seq_len:
            sys.stderr.write(
                'Warning: length {} greather than expected '
                'max length {}\n'.format(length, seq_len)
            )
        yield (curr_idx, curr_idx + length)
        curr_idx += length

def parse_viprbr_r(entry):
    fields = entry.split('|')
    date = fields[5]
    host = fields[6]
    country = fields[8]
    strain = fields[10]
    
    if date == 'NA':
        date = None
    else:
        date = date.split('/')[0]
        date = dparse(date.replace('_', '-'))

    from locations import country2continent
    if country in country2continent:
        continent = country2continent[country]
    else:
        country = 'NA'
        continent = 'NA'

    from mammals import species2group
    if host in species2group.keys():
        group = species2group[host]
    else:
        group = 'NA'
        
    meta = {
    'strain': strain,
    'host': host,
    'group': group,
    'country': country,
    'continent': continent,
    'dataset': 'viprbrc',
    }
    return meta

def featurize_seqs(seqs, vocabulary):
    start_int = len(vocabulary) + 1
    end_int = len(vocabulary) + 2
    sorted_seqs = sorted(seqs.keys())
    X = np.concatenate([
        np.array([ start_int ] + [
            vocabulary[word] for word in seq
        ] + [ end_int ]) for seq in sorted_seqs
    ]).reshape(-1, 1)
    lens = np.array([ len(seq) + 2 for seq in sorted_seqs ])
    assert(sum(lens) == X.shape[0])
    return X, lens



def cross_entropy(logprob, n_samples):
    return -logprob / n_samples

def report_performance(model_name, model, vocabulary,
                       train_seqs, test_seqs):
    X_train, lengths_train = featurize_seqs(train_seqs, vocabulary)
    logprob = model.score(X_train, lengths_train)
    tprint('Model {}, train cross entropy: {}'
           .format(model_name, cross_entropy(logprob, len(lengths_train))))
    X_test, lengths_test = featurize_seqs(test_seqs, vocabulary)
    logprob = model.score(X_test, lengths_test)
    tprint('Model {}, test cross entropy: {}'
           .format(model_name, cross_entropy(logprob, len(lengths_test))))

def batch_train(model, seqs, vocabulary, batch_size=1000,
                verbose=True):
    # Control epochs here.
    n_epochs = 1

    n_batches = math.ceil(len(seqs) / float(batch_size))
    if verbose:
        tprint('Traing seq batch size: {}, N batches: {}'
               .format(batch_size, n_batches))

    for epoch in range(n_epochs):
        if verbose:
            tprint('True epoch {}/{}'.format(epoch + 1, n_epochs))
        perm_seqs = [ str(s) for s in seqs.keys() ]
        np.random.shuffle(perm_seqs)

        for batchi in range(n_batches):
            start = batchi * batch_size
            end = (batchi + 1) * batch_size
            seqs_batch = { seq: seqs[seq] for seq in perm_seqs[start:end] }
            X_batch, lengths_batch = featurize_seqs(seqs_batch, vocabulary)
            model = model.fit(X_batch, lengths_batch)
            del seqs_batch
 
def embed_seqs(model, seqs, vocabulary, virus, use_cache= True, embed_fname = ''):
    X_cat, lengths = featurize_seqs(seqs, vocabulary)
    if use_cache:
        mkdir_p('target/{}/embedding'.format(virus))
        embed_fname = ('target/{}/embedding/{}_{}.npy'
                        .format(virus, model.model_name_, model.hidden_dim_))
    X_embed = model.transform(X_cat, lengths, embed_fname)
    sorted_seqs = sorted(seqs)
   
    for seq_idx, seq in enumerate(sorted_seqs):
        for meta in seqs[seq]:
            meta['embedding'] = X_embed[seq_idx]
    return seqs

def plot_umap(adata, categories, virus='rhabdo'):
    for category in categories:
        plt.figure(figsize = (10,10))
        sc.pl.umap(adata, color=category, size = 100, alpha= 0.8,
                   save='_{}_{}.png'.format(virus, category))


def analyze_embedding(model, seqs, vocabulary, virus):
    seqs = embed_seqs(model, seqs, vocabulary, virus)
    X, obs = [], {}
    obs['n_seq'] = []
    obs['seq'] = []
    for seq in seqs:
        meta = seqs[seq][0]
        X.append(meta['embedding'].mean(0))
        for key in meta:
            if key == 'embedding':
                continue
            if key not in obs:
                obs[key] = []

            if 'vesicul' in meta[key]:
                if 'Chandipura' in meta[key]:
                    obs['family'].append('Vesiculovirus - Chandipura');
                else:
                    obs['family'].append('Vesiculovirus - Other');
            elif 'lyssa' in meta[key]:
                if meta['continent'] != 'NA':
                    obs['family'].append('Lyssavirus - {}'.format(meta['continent'].split()[0]))
                else:
                    obs['family'].append('Lyssavirus - Other')
            obs[key].append(Counter([
                meta[key] for meta in seqs[seq]
            ]).most_common(1)[0][0])
        obs['n_seq'].append(len(seqs[seq]))
        obs['seq'].append(str(seq))
    X = np.array(X)

    adata = AnnData(X)
    adata.write('./target/{}_embed.h5ad'.format(virus))

    for key in obs:
        adata.obs[key] = obs[key]

    sc.pp.neighbors(adata, n_neighbors=20, use_rep='X')
    sc.tl.louvain(adata, resolution=1.)
    sc.tl.umap(adata, min_dist=1.)

    sc.set_figure_params(dpi_save=500)
    plot_umap(adata, [ 'host', 'group', 'strain', 'continent', 'family', 'louvain' ])