"""Interactive local fMRI viewer (NumPy + Matplotlib with a desktop backend).

Run: .venv/bin/python local_visualizer.py [--window-trs 200]
Hover to highlight a row; click to open its eight panels showing the full run.
Click a plot to zoom in or out. While zoomed, scroll or use arrow keys to pan;
the slider and Window field control the shared TR axis. Full run resets the
view, Back returns to the table, and Exit closes all application windows.
Toolbar zoom/pan/save
are also available. Default paths are relative to this script, not the cwd.
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backend_bases import cursors
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
from matplotlib.ticker import MaxNLocator
from matplotlib.widgets import Button, Slider, TextBox

# Logical size for a 3456 × 2234 Retina screen at 2× scaling.
SCREEN_FIGSIZE = (17.28, 11.17)
FIRST_TR = 20

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / 'plots/fMRI_classifier/LORO_participant_class_balanced'
MOTION_RESULTS = ROOT / 'plots/motion_classifier/LORO_participant_class_balanced'
COLORS = {'responsive': '#2b8cbe', 'non-responsive': '#d7301f'}
PANEL_HIGHLIGHT = '#fff9d9'
PROB_CMAP = LinearSegmentedColormap.from_list('probability', list(COLORS.values()))
RECALL_CMAP = LinearSegmentedColormap.from_list('recall', ['#ef8a8a', '#fff1a8', '#87cf95'])


def read_csv(path, required):
    with path.open(newline='', encoding='utf-8-sig') as stream:
        reader = csv.DictReader(stream)
        missing = set(required).difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f'{path}: missing columns: {", ".join(sorted(missing))}')
        return list(reader)


def load_vector(path):
    values = np.asarray(np.load(path, allow_pickle=False), dtype=float).squeeze()
    if values.ndim == 0:
        values = values.reshape(1)
    if values.ndim != 1:
        raise ValueError(f'{path}: expected a vector, got {values.shape}')
    return values


def load_run(raw_dir, participant, run):
    """Load aligned data; rate uses the classifier plot's centered 10-TR window."""
    folder = raw_dir / participant / run
    times = load_vector(folder / 'time_vector.npy')
    if len(times) < 2 or not np.all(np.isfinite(times)) or np.any(np.diff(times) <= 0):
        raise ValueError(f'{folder}: time_vector must contain increasing finite times')
    tr_seconds = float(np.median(np.diff(times)))
    signals = [load_vector(folder / f'mean_z{name}.npy')
               for name in ('AAN', 'Thal', 'Cort', 'BOLD')]
    if any(len(signal) != len(times) for signal in signals):
        raise ValueError(f'{folder}: signal and time_vector lengths differ')
    presses = load_vector(folder / 'bps.npy')
    presses = np.sort(presses[np.isfinite(presses)])
    window = 10 * tr_seconds
    counts = (np.searchsorted(presses, times + window / 2, side='left')
              - np.searchsorted(presses, times - window / 2, side='left'))
    motion_path = folder / 'motion_TRs.npy'
    motion = load_vector(motion_path) if motion_path.exists() else np.array([])
    # Segments and ROI samples use zero-based TR indices.
    return signals, (times - times[0]) / tr_seconds, counts / window, motion, tr_seconds


class RunViewer:
    def __init__(self, raw_dir, participant, run, predictions, window_trs,
                 on_back=None, on_exit=None, fold_results=None, motion_predictions=None,
                 motion_fold_results=None):
        signals, time_trs, rate, motion, _ = load_run(raw_dir, participant, run)
        if len(signals[0]) <= FIRST_TR:
            raise ValueError(f'{participant}/{run}: no data at or after TR {FIRST_TR}')
        self.start = float(FIRST_TR)
        segments = sorted(predictions, key=lambda row: float(row['segment_start_TR']))
        starts = np.array([float(row['segment_start_TR']) for row in segments])
        ends = np.array([float(row['segment_end_TR_exclusive']) for row in segments])
        probs = np.array([float(row['probability_nonresponsive']) for row in segments])
        labels = [row['label'] for row in segments]
        if set(labels).difference(COLORS):
            raise ValueError(f'{participant}/{run}: unknown responsiveness label')
        if (not np.all(np.isfinite(starts)) or not np.all(np.isfinite(ends))
                or np.any(starts < 0) or np.any(ends <= starts)
                or np.any(ends > len(signals[0]))):
            raise ValueError(f'{participant}/{run}: invalid segment boundaries')
        if not np.all(np.isfinite(probs)) or np.any((probs < 0) | (probs > 1)):
            raise ValueError(f'{participant}/{run}: probabilities must be in [0, 1]')
        self.segment_starts, self.segment_ends = starts, ends
        self.segment_probs, self.segment_labels = probs, labels
        motion_by_segment = {}
        for row in motion_predictions or []:
            key = (float(row['segment_start_TR']), float(row['segment_end_TR_exclusive']))
            probability = float(row['probability_nonresponsive'])
            if key in motion_by_segment:
                raise ValueError(f'{participant}/{run}: duplicate motion prediction for {key}')
            if not np.isfinite(probability) or not 0 <= probability <= 1:
                raise ValueError(f'{participant}/{run}: motion probabilities must be in [0, 1]')
            motion_by_segment[key] = probability
        self.segment_motion_probs = np.array([
            motion_by_segment.get((start, end), np.nan) for start, end in zip(starts, ends)])
        self.signals, self.time_trs, self.press_rate = signals, time_trs, rate
        self.motion = motion[motion >= FIRST_TR]
        self._hover_segment = None
        self._overlay_axes = []
        self._overlay_card = None
        self.fig, self.axes = plt.subplots(
            8, 1, sharex=True, figsize=SCREEN_FIGSIZE,
            gridspec_kw={'height_ratios': [1, 1, 1, 1, 1.8, 1.8, 0.8, 1], 'hspace': 0.15})
        self.fig.canvas.manager.set_window_title(f'{participant} | {run}')
        self.fig.subplots_adjust(left=0.12, right=0.985, top=0.88, bottom=0.17)
        recall_lines = []
        for classifier, results in (('fMRI', fold_results), ('Motion LR', motion_fold_results)):
            results = results or {}
            recall_labels = []
            for prefix, label in (('responsive', 'Responsive'), ('nonresponsive', 'Nonresponsive')):
                count = int(results.get(f'n_{prefix}', 0))
                raw = results.get(f'{prefix}_recall', '')
                value = float(raw) if raw else float('nan')
                recall = f'{value:.1%}' if count > 0 and np.isfinite(value) else 'N/A'
                recall_labels.append(f'{label} recall: {recall} (n={count})')
            recall_lines.append(f'{classifier}: ' + '   |   '.join(recall_labels))
        self.fig.suptitle(f'{participant} | {run}\n' + '\n'.join(recall_lines),
                          x=0.56, y=0.99, fontsize=12)
        self.on_back = on_back
        self.back_button = Button(self.fig.add_axes([0.015, 0.94, 0.09, 0.03]),
                                  '← Back to runs')
        self.back_button.label.set_fontsize(9)
        self.back_button.on_clicked(self.go_back)
        self.full_run_button = Button(self.fig.add_axes([0.12, 0.94, 0.075, 0.03]),
                                      'Full run')
        self.full_run_button.on_clicked(self.show_full_run)
        self.exit_button = Button(self.fig.add_axes([0.90, 0.935, 0.075, 0.04]),
                                  'Exit', hovercolor='#efb5b5')
        self.exit_button.on_clicked(on_exit if on_exit is not None else self.go_back)
        for ax, signal, name, color in zip(
                self.axes[:4], signals, ('AAN', 'Thalamus', 'Cortex', 'BOLD'),
                ('#d95f02', '#1b9e77', '#377eb8', '#756bb1')):
            ax.plot(np.arange(FIRST_TR, len(signal)), signal[FIRST_TR:], color=color, linewidth=1.8)
            ax.axhline(0, color='0.4', linewidth=1)
            ax.set_ylabel(f'Mean (z) \n{name}', fontsize=9)
        probability_ax, motion_probability_ax, label_ax, rate_ax = self.axes[4:]
        for ax in (self.axes[3], probability_ax, rate_ax):
            ax.set_facecolor(PANEL_HIGHLIGHT)
        midpoint = (starts + ends) / 2
        visible_predictions = midpoint >= FIRST_TR
        probability_ax.plot(midpoint[visible_predictions], probs[visible_predictions], color='0.65', linewidth=1)
        self._dot_segments = np.flatnonzero(visible_predictions)
        self.prediction_dots = probability_ax.scatter(midpoint[visible_predictions], probs[visible_predictions],
                               c=[COLORS[label] for label, visible in zip(labels, visible_predictions) if visible],
                               s=24, edgecolors='white', linewidths=0.35, zorder=4)
        self.prediction_dots.set_pickradius(8)
        probability_ax.axhline(0.5, color='0.3', linestyle='--', linewidth=1.6)
        probability_ax.set(ylim=(-0.03, 1.03), ylabel='fMRI\nP(NR)')
        motion_probs = self.segment_motion_probs
        motion_probability_ax.plot(midpoint[visible_predictions], motion_probs[visible_predictions],
                                   color='0.65', linewidth=1)
        self.motion_prediction_dots = motion_probability_ax.scatter(
            midpoint[visible_predictions], motion_probs[visible_predictions],
            c=[COLORS[label] for label, visible in zip(labels, visible_predictions) if visible],
            s=24, edgecolors='white', linewidths=0.35, zorder=4)
        self.motion_prediction_dots.set_pickradius(8)
        motion_probability_ax.axhline(0.5, color='0.3', linestyle='--', linewidth=1.6)
        motion_probability_ax.set(ylim=(-0.03, 1.03), ylabel='Motion LR\nP(NR)')
        if np.any(~np.isfinite(motion_probs)) or not len(motion_probs):
            motion_probability_ax.text(0.99, 0.85, 'Missing motion predictions shown as gaps',
                                       transform=motion_probability_ax.transAxes,
                                       ha='right', fontsize=8, color='0.4')
        for start, end, label in zip(starts, ends, labels):
            if end > FIRST_TR:
                label_ax.hlines(int(label == 'non-responsive'), max(start, FIRST_TR), end,
                                color=COLORS[label], linewidth=10)
        label_ax.set_yticks([0, 1], labels=['Responsive', 'Non-responsive'])
        label_ax.set(ylim=(-0.5, 1.5), ylabel='Actual')
        visible_times = time_trs >= FIRST_TR
        rate_ax.plot(time_trs[visible_times], rate[visible_times], color='#238b45', linewidth=2)
        rate_ax.set(xlabel='Time (TR, zero-based)', ylabel='Press rate\n(Hz)')
        rate_ax.set_ylim(bottom=0, top=max(0.1, float(rate[visible_times].max(initial=0)) * 1.1))
        motion = motion[motion >= FIRST_TR]
        for ax in self.axes:
            ax.grid(alpha=0.18, linewidth=1.2)
            if len(motion):
                ax.vlines(motion, 0, 1, transform=ax.get_xaxis_transform(),
                          colors='0.35', linestyles=':', linewidth=1.4, alpha=0.45)
        handles = [Line2D([], [], marker='o', linestyle='none', color=COLORS[label],
                          label=f'Actual: {label}')
                   for label in ('non-responsive', 'responsive')]
        if len(motion):
            handles.append(Line2D([], [], color='0.35', linestyle=':', linewidth=1.4, label='Motion TR'))
        self.fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.55, 0.925),
                        ncol=len(handles), frameon=False)
        pos = probability_ax.get_position()
        colorbar_ax = self.fig.add_axes([pos.x0 - 0.065, pos.y0, 0.012, pos.height])
        bar = self.fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(0, 1), cmap=PROB_CMAP),
                               cax=colorbar_ax)
        bar.set_ticks([0, 0.5, 1])
        bar.ax.yaxis.set_ticks_position('left')
        self.total = float(max(len(signals[0]), time_trs[-1]))
        self.span = self.total - self.start
        self.zoom_width = min(float(window_trs), self.span)
        self.width = self.span
        self.overview = True
        self.slider = Slider(self.fig.add_axes([0.22, 0.08, 0.50, 0.022]),
                             'Start TR', self.start, max(self.total - self.zoom_width, self.start + 1),
                             valinit=self.start)
        self.slider.on_changed(self.pan)
        self.window_box = TextBox(self.fig.add_axes([0.87, 0.071, 0.09, 0.035]),
                                  'Window (TR) ', initial=f'{self.zoom_width:g}')
        self.window_box.on_submit(self.resize_window)
        self.fig.text(0.12, 0.025,
                      'Click a plot: zoom in / out   |   When zoomed: scroll / ← → to pan',
                      fontsize=9, color='0.35')
        self.fig.canvas.mpl_connect('scroll_event', self.on_scroll)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.fig.canvas.mpl_connect('button_press_event', self.on_plot_click)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_segment_hover)
        self.fig.canvas.mpl_connect('figure_leave_event', lambda event: self.hide_segment_overlay())
        self.show_full_run()
        self.fig.canvas.manager.full_screen_toggle()

    def on_segment_hover(self, event):
        if event.inaxes in self._overlay_axes:
            return
        if self._hover_segment is not None and self._overlay_card.contains(event)[0]:
            return
        toolbar = self.fig.canvas.manager.toolbar
        if toolbar is not None and toolbar.mode:
            self.hide_segment_overlay()
            return
        index = None
        if event.inaxes in (self.axes[4], self.axes[5]):
            dots = (self.prediction_dots if event.inaxes == self.axes[4]
                    else self.motion_prediction_dots)
            hit, info = dots.contains(event)
            if hit and len(info['ind']):
                candidates = info['ind']
                points = event.inaxes.transData.transform(dots.get_offsets()[candidates])
                nearest = np.argmin(np.sum((points - [event.x, event.y]) ** 2, axis=1))
                index = int(self._dot_segments[candidates[nearest]])
        elif event.inaxes == self.axes[6] and event.xdata is not None:
            candidates = np.flatnonzero((self.segment_starts <= event.xdata)
                                        & (event.xdata < self.segment_ends))
            if len(candidates):
                ys = np.array([int(self.segment_labels[i] == 'non-responsive') for i in candidates])
                pixels = self.axes[6].transData.transform(np.column_stack(
                    (np.full(len(candidates), event.xdata), ys)))
                distances = np.abs(pixels[:, 1] - event.y)
                nearest = np.argmin(distances)
                if distances[nearest] <= max(8, 6 * self.fig.dpi / 72):
                    index = int(candidates[nearest])
        if index is None:
            self.hide_segment_overlay()
        elif index != self._hover_segment:
            self.show_segment_overlay(index, event.x / self.fig.bbox.width)

    def hide_segment_overlay(self):
        if self._hover_segment is None:
            return
        self._hover_segment = None
        for ax in self._overlay_axes:
            ax.set_visible(False)
        self._overlay_card.set_visible(False)
        self._overlay_title.set_visible(False)
        self.fig.canvas.draw_idle()

    def show_segment_overlay(self, index, cursor_fraction):
        """Preview this CSV segment's exact [start, end) interval, without panning."""
        start = max(self.segment_starts[index], self.start)
        end = self.segment_ends[index]
        if end <= start:
            self.hide_segment_overlay()
            return
        # Put the card opposite the pointer so it never obscures its hover target.
        left = 0.035 if cursor_fraction > 0.56 else 0.535
        if self._overlay_card is None:
            self._overlay_card = FancyBboxPatch(
                (left, 0.16), 0.43, 0.74, boxstyle='round,pad=0.012',
                transform=self.fig.transFigure, facecolor='#fbfcfe',
                edgecolor='#54708e', linewidth=1.5, zorder=20)
            self.fig.add_artist(self._overlay_card)
            self._overlay_title = self.fig.text(0, 0, '', fontsize=11, va='top', zorder=31)
            for i in range(8):
                ax = self.fig.add_axes([0, 0, 0.1, 0.1], zorder=30)
                self._overlay_axes.append(ax)
        self._hover_segment = index
        self._overlay_card.set_x(left)
        self._overlay_card.set_visible(True)
        self._overlay_title.set_position((left + 0.012, 0.885))
        label = self.segment_labels[index]
        self._overlay_title.set_text(
            f'TR {start:g}–{end:g} ({end - start:g} TRs; end exclusive)\n'
            f'Actual: {label}  |  P(nonresponsive): {self.segment_probs[index]:.3f}')
        self._overlay_title.set_visible(True)
        color = COLORS[label]
        names = ('AAN', 'Thalamus', 'Cortex', 'BOLD', 'fMRI P(NR)',
                 'Motion P(NR)', 'Actual', 'Rate (Hz)')
        signal_colors = ('#d95f02', '#1b9e77', '#377eb8', '#756bb1')
        trs = np.arange(len(self.signals[0]))
        selected = (trs >= start) & (trs < end)
        motion = self.motion[(self.motion >= start) & (self.motion < end)]
        for i, ax in enumerate(self._overlay_axes):
            ax.clear()
            ax.set_position([left + 0.085, 0.22 + (7 - i) * 0.075, 0.33, 0.062])
            ax.set_visible(True)
            ax.set_facecolor(PANEL_HIGHLIGHT if i in (3, 4, 7) else '#fbfcfe')
            if i < 4:
                ax.plot(trs[selected], self.signals[i][selected], color=signal_colors[i], linewidth=1.8)
                ax.axhline(0, color='0.4', linewidth=1)
            elif i in (4, 5):
                probability = (self.segment_probs[index] if i == 4
                               else self.segment_motion_probs[index])
                if np.isfinite(probability):
                    ax.scatter([(start + end) / 2], [probability], color=color,
                               s=40, edgecolors='white', linewidths=0.35, zorder=4)
                else:
                    ax.text(0.5, 0.5, 'No motion prediction', transform=ax.transAxes,
                            ha='center', fontsize=8)
                ax.axhline(0.5, color='0.3', linestyle='--', linewidth=1.6)
                ax.set_ylim(-0.03, 1.03)
                ax.set_yticks([0, 0.5, 1])
            elif i == 6:
                ax.hlines(int(label == 'non-responsive'), start, end, color=color, linewidth=10)
                ax.set_yticks([0, 1], labels=['R', 'NR'])
                ax.set_ylim(-0.5, 1.5)
            else:
                ax.plot(trs[selected], self.press_rate[selected],
                        color='#238b45', linewidth=2)
                ax.set_ylim(bottom=0, top=max(0.1, self.press_rate[selected].max(initial=0) * 1.1))
            if len(motion):
                ax.vlines(motion, 0, 1, transform=ax.get_xaxis_transform(),
                          colors='0.35', linestyles=':', linewidth=1.4, alpha=0.45)
            ax.set_xlim(start, end)
            ax.set_ylabel(names[i], fontsize=9)
            ax.grid(alpha=0.18, linewidth=1.2)
            ax.tick_params(labelsize=8, labelbottom=i == 7)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
            if i not in (4, 5, 6):
                ax.yaxis.set_major_locator(MaxNLocator(nbins=2))
        self._overlay_axes[-1].set_xlabel('Time (TR)', fontsize=9)
        self.fig.canvas.draw_idle()

    def show_full_run(self, event=None):
        self.hide_segment_overlay()
        self.overview = True
        self.width = self.span
        self.slider.set_val(self.start)
        self.slider.set_active(False)
        self.slider.ax.set_alpha(0.4)
        self.axes[-1].set_xlim(self.start, self.total)
        self.fig.canvas.draw_idle()

    def zoom_at(self, center):
        self.overview = False
        self.width = self.zoom_width
        self.slider.set_active(True)
        self.slider.ax.set_alpha(1)
        self.slider.set_val(np.clip(center - self.width / 2,
                                    self.start, max(self.start, self.total - self.width)))

    def on_plot_click(self, event):
        toolbar = self.fig.canvas.manager.toolbar
        if (event.inaxes not in self.axes or event.xdata is None
                or (toolbar is not None and toolbar.mode)):
            return
        if event.button == 1:
            if self.overview:
                self.zoom_at(event.xdata)
            else:
                self.show_full_run()
        elif event.button == 3 and not self.overview:
            self.show_full_run()

    def pan(self, start):
        self.hide_segment_overlay()
        if self.overview:
            return
        start = float(np.clip(start, self.start, max(self.start, self.total - self.width)))
        self.axes[-1].set_xlim(start, start + self.width)
        self.fig.canvas.draw_idle()

    def shift(self, direction):
        if self.overview:
            return
        current = self.axes[-1].get_xlim()[0]
        self.slider.set_val(np.clip(current + direction * self.width * 0.15,
                                    self.start, max(self.start, self.total - self.width)))

    def on_scroll(self, event):
        if event.inaxes in self.axes:
            self.shift(event.step)

    def on_key(self, event):
        if self.window_box.capturekeystrokes:
            return
        if event.key == 'escape':
            if self.overview:
                self.go_back()
            else:
                self.show_full_run()
        elif event.key in ('left', 'right'):
            self.shift(1 if event.key == 'right' else -1)

    def go_back(self, event=None):
        plt.close(self.fig)
        if self.on_back is not None:
            self.on_back()

    def resize_window(self, text):
        try:
            width = float(text)
            if not np.isfinite(width) or width <= 0:
                raise ValueError
        except ValueError:
            self.window_box.set_val(f'{self.zoom_width:g}')
            return
        center = sum(self.axes[-1].get_xlim()) / 2
        self.zoom_width = min(width, self.span)
        self.slider.valmax = max(self.total - self.zoom_width, self.start + 1)
        self.slider.ax.set_xlim(self.start, self.slider.valmax)
        if not self.overview:
            self.zoom_at(center)


class RunBrowser:
    TABLE_COLUMNS = 3

    def __init__(self, raw_dir, results_dir, window_trs, motion_results_dir=MOTION_RESULTS):
        self.raw_dir, self.window_trs = raw_dir, window_trs
        self.rows = read_csv(results_dir / 'loro_fold_results.csv', (
            'participant', 'run', 'responsive_recall', 'nonresponsive_recall',
            'n_responsive', 'n_nonresponsive'))
        self.rows.sort(key=lambda row: (row['participant'], row['run']))
        if not self.rows:
            raise ValueError('The fold results CSV contains no participant-runs')
        self.motion_fold_results = {}
        for row in read_csv(motion_results_dir / 'loro_fold_results.csv', (
                'participant', 'run', 'responsive_recall', 'nonresponsive_recall',
                'n_responsive', 'n_nonresponsive')):
            key = row['participant'], row['run']
            if key in self.motion_fold_results:
                raise ValueError(f'Duplicate motion fold results for {key}')
            self.motion_fold_results[key] = row
        self.predictions = {}
        for row in read_csv(results_dir / 'loro_predictions.csv', (
                'participant', 'run', 'label', 'segment_start_TR',
                'segment_end_TR_exclusive', 'probability_nonresponsive')):
            self.predictions.setdefault((row['participant'], row['run']), []).append(row)
        self.motion_predictions = {}
        for row in read_csv(motion_results_dir / 'loro_predictions.csv', (
                'participant', 'run', 'segment_start_TR',
                'segment_end_TR_exclusive', 'probability_nonresponsive')):
            self.motion_predictions.setdefault((row['participant'], row['run']), []).append(row)
        self.viewers = {}
        self.fig = plt.figure(figsize=SCREEN_FIGSIZE)
        self.fig.canvas.manager.set_window_title('Local fMRI visualizer')
        self.fig.text(0.025, 0.95, 'Erin\'s fMRI Viewer ˃ᵕ˂', fontsize=22)
        self.exit_button = Button(self.fig.add_axes([0.90, 0.935, 0.075, 0.04]),
                                  'Exit', hovercolor='#efb5b5')
        self.exit_button.on_clicked(self.close_app)
        self.status = self.fig.text(0.025, 0.035, '', fontsize=10, color='0.3', wrap=True)
        self.tables = []
        self._cell_colors = {}
        self._hovered = None
        self.draw_tables()
        self.fig.canvas.mpl_connect('button_press_event', self.open_row)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_hover)
        self.fig.canvas.mpl_connect('figure_leave_event', lambda event: self.highlight_row(None))
        self.fig.canvas.manager.full_screen_toggle()

    def draw_tables(self):
        rows_per_column = (len(self.rows) + self.TABLE_COLUMNS - 1) // self.TABLE_COLUMNS
        gap, margin = 0.018, 0.025
        width = (1 - 2 * margin - gap * (self.TABLE_COLUMNS - 1)) / self.TABLE_COLUMNS
        for column in range(self.TABLE_COLUMNS):
            offset = column * rows_per_column
            rows = self.rows[offset:offset + rows_per_column]
            if not rows:
                continue
            ax = self.fig.add_axes([margin + column * (width + gap), 0.085, width, 0.79])
            ax.axis('off')
            cells, backgrounds = [], []
            for row in rows:
                text = [f"{row['participant']} / {row['run']}"]
                colors = ['#f4f6f8']
                for prefix in ('responsive', 'nonresponsive'):
                    raw = row[f'{prefix}_recall']
                    value = float(raw) if raw else float('nan')
                    count = int(row[f'n_{prefix}'])
                    available = np.isfinite(value) and count > 0
                    text.append(f'{value:.1%} (n={count})' if available else f'N/A (n={count})')
                    colors.append(RECALL_CMAP(np.clip(value, 0, 1)) if available else '#e3e3e3')
                cells.append(text)
                backgrounds.append(colors)
            height = (len(rows) + 1) / (rows_per_column + 1)
            table = ax.table(
                cellText=cells, cellColours=backgrounds,
                colLabels=['Participant / run', 'Responsive\nrecall', 'Non-responsive\nrecall'],
                colWidths=[0.38, 0.31, 0.31], cellLoc='center',
                bbox=[0, 1 - height, 1, height])
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            for (row, col), cell in table.get_celld().items():
                cell.set_edgecolor('white')
                if row == 0:
                    cell.set_facecolor('#243447')
                    cell.set_text_props(color='white', weight='bold')
                self._cell_colors[column, row, col] = cell.get_facecolor()
            self.tables.append((ax, table, offset))
        self.fig.canvas.set_cursor(cursors.POINTER)
        self.fig.canvas.draw_idle()

    def row_at(self, event):
        for column, (ax, table, offset) in enumerate(self.tables):
            if event.inaxes == ax:
                for (row, col), cell in table.get_celld().items():
                    if row > 0 and cell.contains(event)[0]:
                        return column, row
        return None

    def on_hover(self, event):
        self.highlight_row(self.row_at(event))

    def highlight_row(self, index):
        if index == self._hovered:
            return
        for location, active in ((self._hovered, False), (index, True)):
            if location is None:
                continue
            column, row = location
            table = self.tables[column][1]
            for col in range(3):
                cell = table[row, col]
                base = np.array(self._cell_colors[column, row, col])
                tint = np.array([0.3, 0.6, 0.95, 1])
                cell.set_facecolor(0.82 * base + 0.18 * tint if active else base)
                cell.set_edgecolor('#3478bc' if active else 'white')
                cell.set_linewidth(1.5 if active else 1)
                cell.set_text_props(weight='bold' if active else 'normal')
        self._hovered = index
        self.fig.canvas.set_cursor(cursors.HAND if index is not None else cursors.POINTER)
        self.fig.canvas.draw_idle()

    def show_browser(self):
        # Bring the all-runs overview forward when returning from a run.
        if plt.fignum_exists(self.fig.number):
            plt.figure(self.fig.number)
            self.fig.show()
            self.fig.canvas.draw_idle()

    def close_app(self, event=None):
        for viewer in list(self.viewers.values()):
            plt.close(viewer.fig)
        self.viewers.clear()
        plt.close(self.fig)

    def open_row(self, event):
        index = self.row_at(event)
        if index is None or event.button != 1:
            return
        self.highlight_row(None)
        column, table_row = index
        offset = self.tables[column][2]
        row = self.rows[offset + table_row - 1]
        key = row['participant'], row['run']
        try:
            viewer = self.viewers.get(key)
            if viewer is None or not plt.fignum_exists(viewer.fig.number):
                viewer = RunViewer(self.raw_dir, *key, self.predictions.get(key, []),
                                   self.window_trs, on_back=self.show_browser,
                                   on_exit=self.close_app, fold_results=row,
                                   motion_predictions=self.motion_predictions.get(key, []),
                                   motion_fold_results=self.motion_fold_results.get(key))
                self.viewers[key] = viewer
                viewer.fig.canvas.mpl_connect(
                    'close_event', lambda event, key=key: self.viewers.pop(key, None))
            viewer.fig.show()
            self.status.set_text(f'Opened {key[0]} / {key[1]}' +
                                 (' — no classifier predictions' if key not in self.predictions else ''))
            self.status.set_color('0.3')
        except (OSError, ValueError) as error:
            self.status.set_text(f'Cannot open {key[0]} / {key[1]}: {error}')
            self.status.set_color('#b2182b')
        self.fig.canvas.draw_idle()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-data-dir', type=Path, default=ROOT / 'raw_data')
    parser.add_argument('--results-dir', type=Path, default=RESULTS)
    parser.add_argument('--motion-results-dir', type=Path, default=MOTION_RESULTS)
    parser.add_argument('--window-trs', type=float, default=200,
                        help='Click-to-zoom window in TRs (default: 200)')
    args = parser.parse_args()
    if not np.isfinite(args.window_trs) or args.window_trs <= 0:
        parser.error('--window-trs must be a positive finite number')
    try:
        browser = RunBrowser(args.raw_data_dir, args.results_dir, args.window_trs,
                             args.motion_results_dir)
    except (OSError, ValueError) as error:
        parser.exit(1, f'Cannot load local visualizer: {error}\n')
    plt.show()
    return browser


if __name__ == '__main__':
    main()
