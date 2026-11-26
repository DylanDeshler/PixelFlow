"""
Nested, class-stratified training subsets (the data-scaling axis): for a fraction f, every class keeps its first round(f * n_c)
images in a fixed per-class random order (seeded by the class index), so smaller subsets are contained in larger ones and
every class keeps its share. The row list is pinned to a manifest, {data_dir}/subsets/frac{f}.npz, written on first use and
read (and checked against the labels) afterwards, so every run and every later session trains on exactly the same images.

    rows = subset_rows(data_dir, labels, 0.25)       # sorted dataset rows of the N/4 subset

$ python subsets.py --data_dir=/data/in1k_128 --fracs=1,0.5,0.25,0.125      # write / check the manifests, print the sizes
"""

import os
import hashlib

import numpy as np


def _labels_hash(labels):
    return hashlib.sha1(np.ascontiguousarray(np.asarray(labels, dtype=np.int64)).tobytes()).hexdigest()[:16]


def subset_rows(data_dir, labels, frac, seed=0):
    labels = np.asarray(labels)
    if frac >= 1:
        return np.arange(len(labels))
    path = os.path.join(data_dir, 'subsets', f'frac{frac:g}.npz')
    h = _labels_hash(labels)
    if os.path.exists(path):
        m = np.load(path)
        assert str(m['labels_hash']) == h and int(m['seed']) == seed, f'{path} was built for other labels or another seed'
        return m['rows']
    rows = []
    for c in np.unique(labels):
        members = np.flatnonzero(labels == c)
        order = np.random.default_rng([seed, int(c)]).permutation(len(members)) # one fixed order per class: nested across fractions
        rows.append(members[order[:max(1, int(round(frac * len(members))))]])
    rows = np.sort(np.concatenate(rows))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, rows=rows, frac=frac, seed=seed, labels_hash=h)
    return rows


if __name__ == '__main__':
    import argparse
    from data_jit import MemmapImages
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--fracs', default='1,0.5,0.25,0.125')
    args = parser.parse_args()
    labels = np.asarray(MemmapImages(args.data_dir).labels)
    prev = None
    for f in sorted((float(v) for v in args.fracs.split(',')), reverse=True):
        rows = subset_rows(args.data_dir, labels, f)
        counts = np.bincount(labels[rows], minlength=labels.max() + 1)
        nested = '' if prev is None else f", contained in the previous: {bool(np.isin(rows, prev).all())}"
        print(f"frac {f:g}: {len(rows):,} images, per class {counts.min()}-{counts.max()}{nested}")
        prev = rows
