"""
Progress output for long-running steps, readable in log files (the queues redirect to files and are followed with tail):
  bar(iterable, desc)    tqdm, redrawn at most every few seconds so logs stay small
  with stage('FID'):     prints 'FID ...' on entry and 'FID done in 12.3s' on exit
"""

import sys
import time
from contextlib import contextmanager

from tqdm import tqdm


def bar(iterable=None, desc='', total=None, **kwargs):
    return tqdm(iterable, desc=desc, total=total, mininterval=5.0, file=sys.stderr, **kwargs)


@contextmanager
def stage(desc):
    print(f"{desc} ...", flush=True)
    t0 = time.time()
    yield
    dt = time.time() - t0
    print(f"{desc} done in {dt:.1f}s" if dt < 120 else f"{desc} done in {dt / 60:.1f} min", flush=True)
