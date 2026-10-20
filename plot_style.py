"""
Shared chart styling (validated with the dataviz palette validator on the light surface).

Categorical (runs): fixed order, never cycled. Slots 3-6 sit below 3:1 contrast against the surface and aqua/yellow are
close for tritanopes, so every multi-run chart also uses a distinct marker per run plus a legend / direct labels.
Ordinal series (probe blocks, tile sizes) use one-hue blue steps; heatmaps use sequential (one hue) or diverging scales.
"""

SURFACE, INK, INK_2, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3de'
BLUE, ORANGE = '#2a78d6', '#eb6834'
CATEGORICAL = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300']
MARKERS = ['o', 's', '^', 'D', 'v', 'P']
BLUE_STEPS = {3: ['#86b6ef', '#2a78d6', '#104281'], 2: ['#86b6ef', '#184f95'], 1: ['#2a78d6'],
              4: ['#9ec5f4', '#5598e7', '#256abf', '#104281']}

def ordinal_blues(n):
    if n in BLUE_STEPS:
        return BLUE_STEPS[n]
    import matplotlib.cm as cm
    return [cm.Blues(0.35 + 0.6 * i / max(1, n - 1)) for i in range(n)]

def setup():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    return plt

def style_axis(ax, xlabel='', ylabel='', title=''):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=8)
    ax.set_xlabel(xlabel, color=INK_2, fontsize=9)
    ax.set_ylabel(ylabel, color=INK_2, fontsize=9)
    ax.set_title(title, color=INK, fontsize=10, loc='left')
    for axis in (ax.xaxis, ax.yaxis):
        fmt = axis.get_major_formatter()
        if hasattr(fmt, 'set_useOffset'):
            fmt.set_useOffset(False) # no '+1' offset text above linear axes

def line(ax, x, y, color=BLUE, marker='o', label=None, direct_label=None):
    ax.plot(x, y, '-' + marker, color=color, linewidth=2, markersize=5, markeredgecolor=SURFACE, markeredgewidth=1.5, label=label)
    if direct_label and len(x):
        ax.annotate(direct_label, (x[-1], y[-1]), xytext=(6, 0), textcoords='offset points', color=INK_2, fontsize=8, va='center')

def heatmap_style(ax, im, fig, xlabels, ylabels, title, cbar_label=''):
    ax.set_xticks(range(len(xlabels)), xlabels, rotation=45, ha='right', fontsize=7, color=INK_2)
    ax.set_yticks(range(len(ylabels)), ylabels, fontsize=7, color=INK_2)
    ax.set_title(title, color=INK, fontsize=10, loc='left')
    for side in ax.spines.values():
        side.set_visible(False)
    cb = fig.colorbar(im, ax=ax)
    cb.ax.tick_params(labelsize=7, colors=INK_2)
    cb.outline.set_visible(False)
    if cbar_label:
        cb.set_label(cbar_label, color=INK_2, fontsize=8)

def log_axes(ax, x=True, y=False):
    """
    Log scale with plain-number tick labels (not 2x10^0). Ticks at 1/2/5 per decade, or denser when the data spans
    less than a decade (call after plotting, so the data range is known).
    """
    import numpy as np
    from matplotlib.ticker import LogLocator, FuncFormatter, NullLocator
    for on, axis, setter, lims in ((x, ax.xaxis, ax.set_xscale, ax.dataLim.intervalx), (y, ax.yaxis, ax.set_yscale, ax.dataLim.intervaly)):
        if not on:
            continue
        setter('log')
        lo, hi = (float(v) for v in lims)
        span = np.log10(hi / lo) if lo > 0 and hi > lo else 1.0
        subs = (1.0, 2.0, 5.0) if span >= 1 else (1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0) if span < 0.5 else (1.0, 1.5, 2.0, 3.0, 5.0, 7.0)
        axis.set_major_locator(LogLocator(base=10, subs=subs))
        axis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}'))
        axis.set_minor_locator(NullLocator())

def end_labels(ax, x, series, labels, min_gap=0.06):
    """
    Direct labels at the line ends, nudged apart vertically when two ends are closer than min_gap (axes fraction).
    """
    import numpy as np
    lo, hi = ax.get_ylim()
    ys = [(float(y[-1]) - lo) / (hi - lo) for y in series]
    order = np.argsort(ys)
    placed = []
    for i in order:
        y = ys[i]
        if placed and y - placed[-1] < min_gap:
            y = placed[-1] + min_gap
        placed.append(y)
        ax.annotate(labels[i], xy=(x[-1], series[i][-1]), xytext=(6.0, y), textcoords=('offset points', 'axes fraction'),
                    color=INK_2, fontsize=8, va='center', annotation_clip=False)
