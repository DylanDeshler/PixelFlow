"""
Progress output for long-running steps, readable in log files (the queues redirect to files and are followed with tail):
bar(iterable, desc) is tqdm redrawn at most every few seconds, so logs stay small.
"""

import sys

from tqdm import tqdm


def bar(iterable=None, desc='', total=None, **kwargs):
    # position=0: a bar opened while another is running shares its line instead of moving the cursor (escape codes in logs)
    return tqdm(iterable, desc=desc, total=total, mininterval=5.0, file=sys.stderr, position=0, **kwargs)
