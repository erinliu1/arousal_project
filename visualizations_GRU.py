"""Forest plots of specific GRU configurations and participant bootstrap CIs."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

ROOT = Path('plots/GRU')
ROIS = {'all_rois': 'All ROIs', 'aan_rois': 'AAN ROIs',
        'cortical_rois': 'Cortical ROIs', 'thalamic_rois': 'Thalamic ROIs'}
METRICS = {'swba': 'Segment-weighted\nbalanced accuracy',
           'responsive_recall': 'Responsive recall',
           'nonresponsive_recall': 'Nonresponsive recall'}
COLORS = {16: '#0072B2', 32: '#D55E00', 64: '#009E73'}
STYLES = {0.001: '-', 0.003: '--', 0.01: ':'}
CONFIGS = [(hidden, wd) for hidden in COLORS for wd in STYLES]


def load_results(root=ROOT / 'results'):
    """Read one seed-averaged row per configuration, retaining missing CI bounds."""
    frames = []
    for roi in ROIS:
        path = Path(root) / f'{roi}_summary_specific_bootstrap.csv'
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        if frame.empty:
            continue
        if frame.duplicated(['hidden', 'wd']).any():
            raise ValueError(f'Multiple specific configurations share hidden/WD in {path}')
        frame = frame.assign(roi=roi)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def visualize_gru(results=None):
    """Plot four ROI columns and three metric rows with horizontal percentile CIs."""
    results = load_results() if results is None else results
    if results.empty:
        raise ValueError('No specific GRU summary results found; run GRU.py --summarize.')
    positions = np.arange(len(CONFIGS))
    labels = [f'H={hidden:2d}  WD={wd:<5.3f}' for hidden, wd in CONFIGS]
    with plt.rc_context({'font.size': 26, 'axes.labelsize': 30,
                         'axes.linewidth': 2, 'xtick.labelsize': 22,
                         'ytick.labelsize': 26}):
        fig, axes = plt.subplots(3, 4, figsize=(32, 21), sharex=True, sharey=True)
        fig.subplots_adjust(left=.19, right=.82, bottom=.07, top=.94,
                            hspace=.30, wspace=.19)
        for column, (roi, roi_title) in enumerate(ROIS.items()):
            data = results.loc[results.roi.eq(roi)].set_index(['hidden', 'wd'])
            for row, (metric, title) in enumerate(METRICS.items()):
                ax = axes[row, column]
                for position, (hidden, wd) in enumerate(CONFIGS):
                    if (hidden, wd) not in data.index:
                        continue
                    config = data.loc[(hidden, wd)]
                    mean = config[f'{metric}_seed_mean']
                    lower, upper = config[f'{metric}_ci_lower'], config[f'{metric}_ci_upper']
                    color = COLORS[hidden]
                    # Draw bounds independently; percentile CIs need not contain the mean.
                    if pd.notna(lower) and pd.notna(upper):
                        ax.hlines(position, lower, upper, color=color,
                                  linestyle=STYLES[wd], linewidth=5)
                        ax.plot([lower, upper], [position, position], '|', color=color,
                                markersize=18, markeredgewidth=3.5)
                    if pd.notna(mean):
                        ax.plot(mean, position, 'o', color=color, markersize=14, zorder=3)
                if row == 0:
                    ax.set_title(roi_title, fontsize=34, pad=24)
                    ax.axvline(.5, color='0.4', linestyle='--', linewidth=4, zorder=0)
                if column == 0:
                    ax.set_ylabel(title, labelpad=22)
                ax.set_xlabel({
                    'swba': 'Balanced accuracy',
                    'responsive_recall': 'Responsive recall',
                    'nonresponsive_recall': 'Nonresponsive recall',
                }[metric], fontsize=23)
                ax.set_yticks(positions, labels)
                for tick, (hidden, _) in zip(ax.get_yticklabels(), CONFIGS):
                    tick.set_fontfamily('DejaVu Sans Mono')
                    tick.set_horizontalalignment('right')
                    tick.set_color(COLORS[hidden])
                ax.set_ylim(len(CONFIGS) - .5, -.5)
                ax.set_xlim(0, 1)
                ax.set_xticks(np.linspace(0, 1, 6))
                ax.grid(axis='both', alpha=.32, linewidth=1.5)
                ax.tick_params(axis='both', width=2, length=7)
                ax.tick_params(axis='x', bottom=True, labelbottom=True)
                ax.set_axisbelow(True)
                ax.spines[['top', 'right']].set_visible(False)
        handles = [Line2D([], [], color=color, marker='o', linestyle='none',
                          markersize=10, label=f'Hidden = {hidden}')
                   for hidden, color in COLORS.items()]
        handles += [Line2D([], [], color='0.25', lw=2.8, linestyle=style,
                           label=f'WD = {wd:g}') for wd, style in STYLES.items()]
        fig.legend(handles=handles, loc='center left', bbox_to_anchor=(.835, .5),
                   ncol=1, frameon=False, fontsize=34, labelspacing=1.3,
                   handlelength=2.4, handletextpad=.7, markerscale=1.5)
    return fig, axes
