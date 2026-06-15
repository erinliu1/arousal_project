import os
import numpy as np
import matplotlib.pyplot as plt


from framewise_displacement import framewise_displacement
from load_bp_and_arousals import load_no_bp_intervals

AAN_rois = ['LC', 'DR', 'MnR', 'mRt', 'PAG', 'PnO', 'PTg', 'VTA', 'BasalForebrain', 'POA', 'LH']

cortical_rois = ['caudalanteriorcingulate', 'caudalmiddlefrontal', 'cuneus', 'entorhinal', 'frontalpole', 'fusiform', 'insula', 'isthmuscingulate', 'lateraloccipital', 'lateralorbitofrontal', 'lingual', 'medialorbitofrontal', 'ostralmiddlefrontal', 'paracentral', 'parahippocampal', 'pericalcarine', 'postcentral', 'posteriorcingulate', 'precentral', 'precuneus', 'rostralanteriorcingulate', 'superiorfrontal', 'superiorparietal', 'temporalpole']

thalamic_rois = ['AV', 'CM', 'LD', 'LGN', 'MD', 'PUL', 'VA', 'VLa', 'VLp', 'VPL']

directory = '/orcd/data/ldlewis/001/om/erinliu/Arousal_Project'
nicks_dir = os.path.join(directory, 'nicks_data')

my_dir = os.path.join(directory, 'data')

# for participant in sorted(os.listdir(my_dir)):
    # for run in sorted(os.listdir(os.path.join(my_dir, participant))):

participant = 'AAN_08'
run = 'run04'

my_run_path = os.path.join(my_dir, participant, run)
nicks_run_path = os.path.join(nicks_dir, participant, run)
cortex = np.loadtxt(os.path.join(nicks_run_path, 'cortex.txt'))

fd = framewise_displacement(participant, run)
fd = np.asarray(fd).astype(float)
fd_threshold = 0.5
high_motion = fd >= fd_threshold

bps = np.load(os.path.join(my_run_path, 'bps.npy'))
bps = np.asarray(bps).astype(float)
no_bp_intervals = load_no_bp_intervals(bps, seconds_between_bps=20)

time_vector = np.load(os.path.join(my_run_path, 'time_vector.npy'))
time_vector = np.asarray(time_vector).astype(float)
nTRs = len(time_vector)
run_duration_min = (time_vector[-1] - time_vector[0]) / 60
fd_times = (time_vector[:-1] + time_vector[1:]) / 2 # plot at the midpoint between TRs

reference_nTRs = 1000
reference_width = 100
width = reference_width * nTRs / reference_nTRs

fig, ax = plt.subplots(figsize=(width, 9))
ax2 = ax.twinx()

ax.plot(fd_times, fd, marker='o', markersize=1.5, linewidth=1, label='Framewise displacement (mm)')
ax.axhline(fd_threshold, color='orange', linestyle='--', label=f'Motion threshold ({fd_threshold} mm)')
start_idx = None
for i in range(len(high_motion)):
    if high_motion[i] and start_idx is None:
        start_idx = i
    elif not high_motion[i] and start_idx is not None:
        end_idx = i - 1
        ax.axvspan(fd_times[start_idx] - 0.49, fd_times[end_idx] + 0.49, color='lightgray', alpha=0.3, zorder=0)
        start_idx = None

ax.scatter(bps, np.zeros_like(bps), marker='|', color='green', s=120, linewidths=1, label='Button presses')

for sleep_onset_time, arousal_time in no_bp_intervals:
    ax.axvspan(sleep_onset_time, arousal_time, color='lightblue', alpha=0.2, zorder=0)

ax2.plot(time_vector, cortex, color='red', linewidth=1, alpha=0.7, label='average cortical signal')
ax2.set_ylabel('average cortical signal')

ax.set_xticks(time_vector)
labels = []
for i, t in enumerate(time_vector):
    if i % 30 == 0:
        labels.append(f"{t:.2f}")
    else:
        labels.append("")
ax.set_xticklabels(labels)
ax.set_xlabel(f'TR = 0.98s ({run_duration_min:.1f} min)')
ax.set_ylabel('framewise displacement (mm)')

# ax.legend()
ax.grid(axis='x', color='lightgray', alpha=0.2)

for t in time_vector[::30]:
    ax.axvline(t, color='gray', linewidth=1, alpha=0.8, zorder=0)

ymin = min(-0.02, np.min(fd) - 0.02)
ymax = np.max(fd) + 0.05
ax.set_ylim(ymin, ymax)
ax.margins(x=0)
ax.set_xlim(time_vector[0], time_vector[-1])

plt.tight_layout()
plt.savefig('fd_and_bps.png', dpi=300)