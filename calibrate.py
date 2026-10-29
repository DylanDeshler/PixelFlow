"""
The from-scratch calibration procedure for a new dataset: noise schedule and loss weighting found by a procedure, not
inherited from ImageNet. Runs the existing tools in order and records everything in {calib_dir}/calibration.json; steps
already done are skipped on a rerun (delete their entry, or pass --redo=step1,step2, to repeat them).

  stats    data only, seconds (noise_calibration.py): the data's spectrum and the two Gaussian-model transition densities
           over noise levels, coarse structure (R^2, power weighted) and fine detail (information, count weighted). Their
           logit-normal fits bracket the training distribution; they don't determine it (on ImageNet-256 the tuned JiT
           P_mean -0.8 sits ~30% of the way from the R^2 fit, -3.0, to the information fit, +4.3). The starting point is
           taken at that same relative position, and P_std at JiT's ratio to the fitted widths (0.8 / ~1.8) -- a prior
           from one known-good case, which the next steps check.
  warmup   a short training run at the starting point, with the loss weighting being calibrated (train_jit.py)
  snr      gradient signal vs noise per noise level on the warm-up checkpoint's raw weights (analyze_grad_snr.py): which
           noise levels give the optimizer usable gradients at the training batch. Reported, and used to flag sweep
           candidates that put much of their training on noise-dominated levels
  sweep    short runs at sweep_points P_mean values across the bracket around the starting point (same budget each),
           judged on FID -- the final arbiter, since the spectrum only brackets and the gradient signal isn't FID
  sampler  inference only (eval_jit.py), on the best sweep run: time schedules and samplers, judged on FID
Then prints the recommended train_jit.py flags. The noise scale is fixed (noise_scale; 0 -> img_size / 256) and P_mean is
calibrated at it: for noise levels the two are interchangeable (noise_calibration.py).

$ python calibrate.py --data_dir=/data/toy_gauss_128 --train_args="config/toy_gauss_128.py --optimizer_type=muon"
$ python calibrate.py --data_dir=/data/in1k_128 --train_args="config/jit_in1k_128.py --optimizer_type=muon" \\
      --loss_freq_weight=wiener --loss_t_weight=learned --calib_dir=runs/calibration/in1k_128_wiener_learned
$ python calibrate.py ... --steps=stats --dry_run=True      # just the data analysis, and the commands the rest would run
For the plain / Wiener x fixed / learned comparison, run it once per combination with its own --calib_dir.
"""

import os
import re
import sys
import json
import math
import shlex
import subprocess

import numpy as np
import torch

import noise_calibration as nc

# -----------------------------------------------------------------------------
data_dir = '' # the prepared dataset (train split; val/ is used for the gradient analysis)
train_args = '' # train_jit.py's base arguments: config files and flags, e.g. "config/toy_gauss_128.py --optimizer_type=muon"
calib_dir = '' # '' -> runs/calibration/{dataset}_{loss_freq_weight}_{loss_t_weight}
steps = 'stats,warmup,snr,sweep,sampler'
redo = '' # comma list of steps to repeat even if recorded
noise_scale = 0.0 # fixed during calibration; 0 -> img_size / 256 (JiT's rule)
loss_freq_weight = 'none' # the loss weighting being calibrated (train_jit.py flags)
loss_t_weight = 'none'
prior_position = 0.3 # starting P_mean: this far from the R^2 fit to the information fit (JiT's ImageNet-256 position)
prior_std_ratio = 0.45 # starting P_std: this times the fits' mean width (JiT: 0.8 / ~1.8)
max_samples = 5000 # images for the data statistics
warmup_epochs = 5
snr_points = 10 # noise levels for the gradient analysis, evenly spaced in logit(t) over the bracket's central range
snr_batches = 16
noisy_fraction = 0.25 # a noise level is "noise-dominated" when its gradient signal fraction at the training batch is below this
sweep_points = 3
sweep_width = 0.0 # P_mean spacing of the sweep; 0 -> a quarter of the bracket (information fit - R^2 fit)
sweep_epochs = 10
fid_num_images = 10000
samplers = 'heun:50:uniform,heun:50:shift,heun:50:train_quantile,ancestral:100:uniform' # method:steps:time_schedule
sampler_num_images = 10000
python = sys.executable
dry_run = False # print the commands without running them
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'configurator.py')).read())
config = {k: globals()[k] for k in config_keys}
# -----------------------------------------------------------------------------

assert data_dir, 'pass --data_dir'
here = os.path.dirname(os.path.abspath(__file__))
dataset = os.path.basename(os.path.normpath(data_dir))
with open(os.path.join(data_dir, 'meta.json')) as f:
    img_size = json.load(f)['img_size']
noise_scale = noise_scale or img_size / 256
calib_dir = calib_dir or os.path.join('runs', 'calibration', f'{dataset}_{loss_freq_weight}_{loss_t_weight}')
os.makedirs(calib_dir, exist_ok=True)
state_path = os.path.join(calib_dir, 'calibration.json')
state = json.load(open(state_path)) if os.path.exists(state_path) else {}
state['config'] = config
todo = [s for s in steps.split(',') if s]
redo_set = {s for s in redo.split(',') if s}
assert all(s in ('stats', 'warmup', 'snr', 'sweep', 'sampler') for s in todo), f'unknown step in {steps}'

def save(force=False):
    # a dry run records only the data statistics (computed for real); the training / eval steps it only prints
    if dry_run and not force:
        return
    with open(state_path, 'w') as f:
        json.dump(state, f, indent=1)

def should_run(step):
    return step in todo and (step not in state or step in redo_set)

def run(cmd, log_name):
    # runs a command, teeing its output to {calib_dir}/{log_name}.log; returns the output (None in a dry run)
    print('\n$ ' + ' '.join(shlex.quote(c) for c in cmd), flush=True)
    if dry_run:
        return None
    lines = []
    with open(os.path.join(calib_dir, f'{log_name}.log'), 'w') as log:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=here)
        for line in p.stdout:
            log.write(line); lines.append(line)
            if 'FID-' in line or 'error' in line.lower() or line.startswith('Traceback'):
                print('  ' + line.rstrip(), flush=True)
        p.wait()
    out = ''.join(lines)
    if p.returncode != 0:
        raise RuntimeError(f'{cmd[1]} failed (exit {p.returncode}); see {os.path.join(calib_dir, log_name + ".log")}')
    return out

def last_fid(output):
    # the last "FID-10k 83.08" (train_jit.py) / "FID-10k 83.080" (eval_jit.py) in a tool's output
    m = re.findall(r'FID-[\d.]+k ([\d.]+)', output or '')
    return float(m[-1]) if m else None

def train_cmd(out_dir, epochs, P_mean, P_std):
    return [python, 'train_jit.py', *shlex.split(train_args), f'--data_dir={data_dir}', f'--out_dir={out_dir}',
            f'--epochs={epochs}', f'--eval_every_epochs={epochs}', '--warmup_epochs=1', f'--P_mean={P_mean:.4f}', f'--P_std={P_std:.4f}',
            f'--noise_scale={noise_scale}', f'--loss_freq_weight={loss_freq_weight}', f'--loss_t_weight={loss_t_weight}',
            f'--fid_num_images={fid_num_images}', '--wandb_log=False', '--diag=False']

logit = lambda t: math.log(t / (1 - t))

# -----------------------------------------------------------------------------
# 1. data statistics: the bracket and the starting point

if should_run('stats'):
    x = nc.load_samples(data_dir, max_samples)
    res = nc.analyze(dataset, x, noise_scale, -0.8, 0.8, 8, 2.0) # prints the curves' summary (the -0.8 / 0.8 reference is JiT's)
    r2_mean, r2_std = nc.fit_logit_normal(res['r2'], res['u'], noise_scale)
    info_mean, info_std = nc.fit_logit_normal(res['info'], res['u'], noise_scale)
    start_mean = r2_mean + prior_position * (info_mean - r2_mean)
    start_std = prior_std_ratio * 0.5 * (r2_std + info_std)
    state['stats'] = {'noise_scale': noise_scale, 'r2_fit': [r2_mean, r2_std], 'info_fit': [info_mean, info_std],
                      'start': [start_mean, start_std], 'prior_position': prior_position, 'prior_std_ratio': prior_std_ratio}
    save(force=True)
    print(f"\nbracket at noise scale {noise_scale}: P_mean {r2_mean:+.2f} (coarse structure) .. {info_mean:+.2f} (fine detail); "
          f"starting point P_mean {start_mean:+.2f}, P_std {start_std:.2f}")
assert 'stats' in state or dry_run, 'run the stats step first'
st = state.get('stats', {'r2_fit': [-3.0, 1.8], 'info_fit': [4.3, 1.8], 'start': [-0.8, 0.8]})
start_mean, start_std = st['start']

# -----------------------------------------------------------------------------
# 2. warm-up run

warm_dir = os.path.join(calib_dir, 'warmup')
if should_run('warmup'):
    out = run(train_cmd(warm_dir, warmup_epochs, start_mean, start_std), 'warmup')
    state['warmup'] = {'out_dir': warm_dir, 'epochs': warmup_epochs, 'P_mean': start_mean, 'P_std': start_std, 'fid': last_fid(out)}
    save()

# -----------------------------------------------------------------------------
# 3. gradient signal per noise level on the warm-up checkpoint

def noisy_mass(P_mean, P_std, ts, frac):
    # share of a logit-normal training distribution on noise levels whose signal fraction is below noisy_fraction
    # (fractions interpolated in logit(t); outside the measured range, the nearest measurement)
    lts = np.array([logit(t) for t in ts]); f = np.array(frac)
    z = np.random.default_rng(0).normal(P_mean, P_std, 200000)
    return float((np.interp(z, lts, f) < noisy_fraction).mean())

if should_run('snr'):
    lo, hi = sorted((st['r2_fit'][0], st['info_fit'][0]))
    grid = np.linspace(min(lo, start_mean - 2 * start_std), max(hi, start_mean + 2 * start_std), snr_points)
    ts = [float(1 / (1 + math.exp(-g))) for g in grid]
    ts = sorted({min(max(t, 0.01), 0.99) for t in ts})
    out = run([python, 'analyze_grad_snr.py', f'--ckpts={warm_dir}', f'--data_dir={data_dir}', '--ema=0',
               f"--ts={','.join(f'{t:.4f}' for t in ts)}", f'--n_batches={snr_batches}'], 'snr')
    if not dry_run:
        diag = os.path.join(warm_dir, 'diagnostics')
        path = max((os.path.join(diag, f) for f in os.listdir(diag) if f.startswith('grad_snr_') and f.endswith('_raw.json')), key=os.path.getmtime)
        g = json.load(open(path))
        col = g['names'].index('wiener' if loss_freq_weight == 'wiener' else 'plain')
        frac = [row[col] for row in g['frac_at_B']]
        state['snr'] = {'json': path, 't': g['t'], 'signal_fraction': frac, 'B_noise': [row[col] for row in g['B_noise']],
                        'loss': g['names'][col], 'start_noisy_mass': noisy_mass(start_mean, start_std, g['t'], frac)}
        save()
        print(f"\ngradient signal fraction at the training batch, {g['names'][col]} loss (noise-dominated below {noisy_fraction}):")
        for t, f_, b in zip(g['t'], frac, state['snr']['B_noise']):
            print(f"  t={t:.3f} (logit {logit(t):+.2f}): {f_:.2f}  (B_noise {b:.0f})")
        print(f"starting point puts {state['snr']['start_noisy_mass']:.0%} of training on noise-dominated levels")

# -----------------------------------------------------------------------------
# 4. P_mean sweep, judged on FID

if should_run('sweep'):
    width = sweep_width or abs(st['info_fit'][0] - st['r2_fit'][0]) / 4
    means = [start_mean + width * k for k in np.linspace(-1, 1, sweep_points)] if sweep_points > 1 else [start_mean]
    results = []
    for m in means:
        d = os.path.join(calib_dir, f'sweep_pmean{m:+.2f}')
        out = run(train_cmd(d, sweep_epochs, m, start_std), f'sweep_pmean{m:+.2f}')
        r = {'P_mean': m, 'P_std': start_std, 'out_dir': d, 'fid': last_fid(out)}
        if 'snr' in state:
            r['noisy_mass'] = noisy_mass(m, start_std, state['snr']['t'], state['snr']['signal_fraction'])
        results.append(r)
        state['sweep'] = {'width': width, 'epochs': sweep_epochs, 'runs': results, 'done': False}
        save()
    done = [r for r in results if r['fid'] is not None]
    best = min(done, key=lambda r: r['fid']) if done else None
    state['sweep'].update(done=True, best=best)
    save()
    if not dry_run:
        print(f"\nP_mean sweep ({sweep_epochs} epochs each, P_std {start_std:.2f}):")
        for r in results:
            nm = f", {r['noisy_mass']:.0%} of training on noise-dominated levels" if 'noisy_mass' in r else ''
            print(f"  P_mean {r['P_mean']:+.2f}: FID {r['fid']}{nm}" + ('   <- best' if r is best else ''))
    if best is not None and best in (done[0], done[-1]) and len(done) > 1:
        print("  the best is at the edge of the sweep: extend it (--redo=sweep with a shifted --prior_position or a wider --sweep_width)")

# -----------------------------------------------------------------------------
# 5. sampler, inference only

if should_run('sampler'):
    best = state.get('sweep', {}).get('best')
    assert best is not None or dry_run, 'run the sweep step first'
    ckpt = os.path.join(best['out_dir'] if best else os.path.join(calib_dir, 'sweep_best'), 'ckpt.pt')
    results = []
    for spec in samplers.split(','):
        method, n_steps, schedule = spec.split(':')
        cmd = [python, 'eval_jit.py', f'--ckpt={ckpt}', f'--data_dir={data_dir}', '--cfgs=1.0', f'--final_num_images={sampler_num_images}',
               f'--sampling_method={method}', f'--num_sampling_steps={n_steps}', f'--time_schedule={schedule}', '--plot=False']
        out = run(cmd, f'sampler_{method}{n_steps}_{schedule}')
        results.append({'method': method, 'steps': int(n_steps), 'time_schedule': schedule, 'fid': last_fid(out)})
        state['sampler'] = {'ckpt': ckpt, 'runs': results}
        save()
    done = [r for r in results if r['fid'] is not None]
    state['sampler']['best'] = min(done, key=lambda r: r['fid']) if done else None
    save()
    if not dry_run:
        print(f"\nsamplers on {ckpt} (FID-{sampler_num_images / 1000:g}k, cfg 1):")
        for r in results:
            print(f"  {r['method']} {r['steps']} steps, {r['time_schedule']}: FID {r['fid']}")

# -----------------------------------------------------------------------------
# recommendation

if not dry_run:
    best = state.get('sweep', {}).get('best')
    P_mean, P_std = (best['P_mean'], best['P_std']) if best else (start_mean, start_std)
    flags = [f'--noise_scale={noise_scale}', f'--P_mean={P_mean:.2f}', f'--P_std={P_std:.2f}',
             f'--loss_freq_weight={loss_freq_weight}', f'--loss_t_weight={loss_t_weight}']
    smp = state.get('sampler', {}).get('best')
    if smp:
        flags += [f"--sampling_method={smp['method']}", f"--num_sampling_steps={smp['steps']}", f"--time_schedule={smp['time_schedule']}"]
    state['recommended_flags'] = flags
    save()
    print(f"\n{'calibrated' if best else 'starting-point'} settings for {dataset} (state: {state_path}):\n  {' '.join(flags)}")
