"""
Stage a dataset's image arrays on a fast local disk for the length of a queue, and remove them afterwards. Training reads them
through data_jit.images_path ($JIT_FAST_DATA/{absolute data_dir}/images.u8, used only when complete), so staging is only
ever an optimisation: if there is not enough space, the copy fails or the disk disappears, everything reads the original.

  stage    copy {data_dir}/images.u8 and {data_dir}/val/images.u8 to {fast_dir}/{absolute path}/, if the fast disk's free
           space exceeds their size plus --margin_gb; each file is written to a .partial name and renamed only once complete
           (so a half-written copy is never used). Already-staged complete copies are kept. Exit code 0 always: a failure is
           reported and the original files are used
  cleanup  delete everything under {fast_dir} that stage created (the directory has a marker file; nothing else is touched)

$ python fast_data.py stage --data_dir=/data/in1k_128 --fast_dir=/opt/dlami/nvme/jit_fast
$ python fast_data.py cleanup --fast_dir=/opt/dlami/nvme/jit_fast
"""

import os
import sys
import shutil
import argparse

MARKER = '.fast_data_staged'


def stage(data_dir, fast_dir, margin_gb):
    files = [os.path.join(d, 'images.u8') for d in (data_dir, os.path.join(data_dir, 'val')) if os.path.exists(os.path.join(d, 'images.u8'))]
    files = [os.path.realpath(f) if os.path.islink(f) else f for f in files]
    targets = [(f, os.path.join(fast_dir, os.path.abspath(os.path.dirname(f)).lstrip(os.sep), 'images.u8')) for f in files]
    todo = [(s, t) for s, t in targets if not (os.path.exists(t) and os.path.getsize(t) == os.path.getsize(s))]
    if not todo:
        print(f'fast data: already staged in {fast_dir}' if targets else f'fast data: no images.u8 under {data_dir}')
        return
    parent = fast_dir
    while not os.path.exists(parent):
        parent = os.path.dirname(parent)
    need = sum(os.path.getsize(s) for s, _ in todo)
    free = shutil.disk_usage(parent).free
    if free < need + margin_gb * 1e9:
        print(f'fast data: not staged ({need / 1e9:.1f} GB needed + {margin_gb} GB margin, {free / 1e9:.1f} GB free on {parent}); reading {data_dir}')
        return
    os.makedirs(fast_dir, exist_ok=True)
    open(os.path.join(fast_dir, MARKER), 'w').close()
    from util.progress import bar
    for src, dst in todo:
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        tmp = dst + '.partial'
        size, chunk = os.path.getsize(src), 64 << 20
        with open(src, 'rb') as fi, open(tmp, 'wb') as fo:
            for _ in bar(range(-(-size // chunk)), f'fast data: {src} -> {dst} ({size / 1e9:.1f} GB)'):
                fo.write(fi.read(chunk))
        if os.path.getsize(tmp) != size:
            raise IOError(f'{tmp}: {os.path.getsize(tmp)} bytes written, {size} expected')
        os.replace(tmp, dst)
    print(f'fast data: staged {need / 1e9:.1f} GB in {fast_dir}')


def cleanup(fast_dir):
    if not os.path.exists(os.path.join(fast_dir, MARKER)):
        if os.path.exists(fast_dir):
            print(f'fast data: {fast_dir} has no {MARKER} marker; not deleting it')
        return
    shutil.rmtree(fast_dir, ignore_errors=True)
    print(f'fast data: removed {fast_dir}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=('stage', 'cleanup'))
    p.add_argument('--data_dir', default='')
    p.add_argument('--fast_dir', required=True)
    p.add_argument('--margin_gb', type=float, default=10.0)
    a = p.parse_args()
    try:
        stage(a.data_dir, a.fast_dir, a.margin_gb) if a.action == 'stage' else cleanup(a.fast_dir)
    except Exception as e: # never fatal: staging is an optimisation
        print(f'fast data: {a.action} failed ({type(e).__name__}: {e}); continuing with the original files')
        if a.action == 'stage': # drop any partial copy
            for root, _, fs in os.walk(a.fast_dir):
                for f in fs:
                    if f.endswith('.partial'):
                        try:
                            os.remove(os.path.join(root, f))
                        except OSError:
                            pass
    sys.exit(0)
