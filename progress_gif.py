"""
Animated GIFs of the per-eval sample grids, to see training progress for the same seed and classes.

train_jit.py saves samples/samples_{iter}.png (CFG 1) and samples/samples_{iter}_cfg{g}.png at every eval with fixed
noise and classes; this stitches each series into samples/progress.gif and samples/progress_cfg{g}.gif, one frame per
eval, captioned with the iteration (and epoch when config.json is available). train_jit.py rebuilds them after each
eval; for an existing run:
$ python progress_gif.py runs/in100_128/jit-b16/base
"""

import os
import re
import json
import glob
import argparse

import numpy as np
from PIL import Image, ImageDraw


def find_series(run_dir):
    """
    {suffix: [(iter, path), ...]} for the sample grids of a run ('' = CFG 1, '_cfg3' = CFG 3, ...). Looks in samples/
    and, for runs made before samples/ existed, the run dir itself.
    """
    series = {}
    for d in (os.path.join(run_dir, 'samples'), run_dir):
        for path in glob.glob(os.path.join(d, 'samples_*.png')):
            m = re.match(r'samples_(\d+)(_cfg[\d.]+)?\.png$', os.path.basename(path))
            if m:
                series.setdefault(m.group(2) or '', {})[int(m.group(1))] = path # samples/ wins over an older top-level copy
    return {k: sorted(v.items()) for k, v in series.items()}

def iters_per_epoch(run_dir):
    path = os.path.join(run_dir, 'config.json')
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f).get('resolved', {}).get('iters_per_epoch')
    return None

def make_gif(frames, out_path, ipe=None, title='', max_frames=60, frame_ms=600, last_frame_ms=2000):
    """
    frames: [(iter, png path)] in training order. Writes an animated GIF with an iteration / epoch caption on each frame.
    """
    if len(frames) > max_frames: # evenly subsample, always keeping the latest frame
        keep = np.unique(np.linspace(0, len(frames) - 1, max_frames).round().astype(int))
        frames = [frames[i] for i in keep]
    with Image.open(frames[-1][1]) as img:
        size = img.size # every frame at the latest grid's size (older runs may have saved grids at other sizes)
    images = []
    for it, path in frames:
        with Image.open(path) as img:
            grid = img.convert('RGB')
        if grid.size != size:
            grid = grid.resize(size, Image.LANCZOS)
        caption = f"{title}  iter {it:,}" + (f" (epoch {it / ipe:.1f})" if ipe else '')
        bar = 22
        frame = Image.new('RGB', (grid.width, grid.height + bar), 'white')
        frame.paste(grid, (0, bar))
        ImageDraw.Draw(frame).text((6, 5), caption.strip(), fill='black')
        # per-frame adaptive palette: photographic grids band badly with a single shared palette
        images.append(frame.quantize(colors=256, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.FLOYDSTEINBERG))
    durations = [frame_ms] * (len(images) - 1) + [last_frame_ms]
    images[0].save(out_path, save_all=True, append_images=images[1:], duration=durations, loop=0, optimize=False)
    return out_path

def make_progress_gifs(run_dir, max_frames=60):
    """
    Writes samples/progress{suffix}.gif for every sample-grid series with at least two frames; returns {suffix: path}.
    """
    out_dir = os.path.join(run_dir, 'samples')
    os.makedirs(out_dir, exist_ok=True)
    ipe = iters_per_epoch(run_dir)
    gifs = {}
    for suffix, frames in find_series(run_dir).items():
        if len(frames) < 2:
            continue
        title = f"CFG {suffix[4:]}" if suffix else 'CFG 1'
        gifs[suffix] = make_gif(frames, os.path.join(out_dir, f'progress{suffix}.gif'), ipe, title, max_frames)
    return gifs

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('runs', nargs='+', help='run directories')
    parser.add_argument('--max_frames', type=int, default=60)
    args = parser.parse_args()
    for run in args.runs:
        gifs = make_progress_gifs(run, args.max_frames)
        if not gifs:
            print(f"{run}: fewer than two sample grids per series, nothing to animate")
        for suffix, path in gifs.items():
            print(f"wrote {path} ({len(find_series(run)[suffix])} evals)")
