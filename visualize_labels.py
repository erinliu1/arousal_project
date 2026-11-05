"""Plot button presses, behavioral intervals, and GRU segments by participant."""

import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
import numpy as np
import pandas as pd


DEFAULT_SEGMENT_DIR = Path("plots/GRU/segments/R(W=30,S=1,N_min=8,G_max=4):NR(T_min=20):Segment(TR=40,SW=40)")
RAW_DATA_DIR = Path("raw_data")
COLORS = {"responsive": "#59A14F", "non-responsive": "#E15759"}
NAME_PATTERN = re.compile(
    r"R\(W=(?P<W>\d+),S=(?P<S>\d+),N_min=(?P<N>\d+),G_max=(?P<G>\d+)\):"
    r"NR\(T_min=(?P<T>\d+)\):Segment\(TR=(?P<TR>\d+),SW=(?P<SW>\d+)(?P<trim>,trim)?\)"
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--segment-dir", type=Path, default=DEFAULT_SEGMENT_DIR,
                        help="GRU segment folder containing all_segment_metadata.csv")
    parser.add_argument("--pixels-per-tr", type=float, default=3,
                        help="Horizontal resolution allocated to every TR")
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def numeric_key(value):
    match = re.search(r"(\d+)$", str(value))
    return (int(match.group(1)) if match else float("inf"), str(value))


def read_csv(path, required):
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    data = pd.read_csv(path)
    if missing := set(required).difference(data.columns):
        raise ValueError(f"{path} is missing columns: {', '.join(sorted(missing))}")
    return data


def load_inputs(segment_dir):
    match = NAME_PATTERN.fullmatch(segment_dir.name)
    if not match:
        raise ValueError(f"Could not parse GRU settings from folder name: {segment_dir.name}")
    settings = match.groupdict()
    metadata = read_csv(segment_dir / "all_segment_metadata.csv",
                        {"participant", "run", "label", "segment_start_TR", "segment_end_TR_exclusive"})
    response_root = Path("plots/GRU/responsiveness") / (
        f"W={settings['W']},S={settings['S']},N_min={settings['N']},G_max={settings['G']}")
    response_dir = response_root / f"TR={settings['TR']},SW={settings['SW']},trim" if settings["trim"] else response_root
    responsive = read_csv(response_dir / "responsive_intervals.csv", {"participant", "run", "start_TR", "end_TR"})
    nr_dir = Path("plots/GRU/non_responsiveness") / f"T_min={settings['T']}"
    nr_files = ["non_responsive_intervals.csv", "non_responsive_end_intervals.csv", "non_responsive_whole_runs.csv"]
    nonresponsive = pd.concat([read_csv(nr_dir / name, {"participant", "run", "start_TR", "end_TR"})
                               for name in nr_files], ignore_index=True)
    return metadata, responsive, nonresponsive


def participant_runs(participant):
    participant_dir = RAW_DATA_DIR / participant
    runs = []
    for run_dir in sorted((path for path in participant_dir.iterdir() if path.is_dir()), key=lambda path: numeric_key(path.name)):
        time_path, bp_path = run_dir / "time_vector.npy", run_dir / "bp_TRs.npy"
        if time_path.exists() and bp_path.exists():
            time_vector = np.asarray(np.load(time_path), dtype=float)
            n_trs = len(time_vector)
            presses = np.unique(np.asarray(np.load(bp_path), dtype=int))
            runs.append((run_dir.name, time_vector, presses[(presses >= 0) & (presses < n_trs)]))
    return runs


def draw_intervals(ax, data, participant, run, offset, color, alpha):
    selected = data[(data["participant"] == participant) & (data["run"] == run)]
    for row in selected.itertuples(index=False):
        ax.axvspan(offset + row.start_TR, offset + row.end_TR, color=color, alpha=alpha,
                   linewidth=0, zorder=1)


def plot_run(participant, run_data, metadata, responsive, nonresponsive, output_dir, pixels_per_tr, dpi):
    run, time_vector, presses = run_data
    total_trs = len(time_vector)
    width = max(10, total_trs * pixels_per_tr / dpi)
    fig, ax = plt.subplots(figsize=(width, 4.0))
    major_ticks, major_labels, minor_ticks = [], [], []
    elapsed = time_vector - time_vector[0]
    tick_seconds = np.arange(0, elapsed[-1] + 1e-9, 30)
    tick_trs = np.searchsorted(elapsed, tick_seconds, side="left").clip(max=total_trs - 1)
    for seconds, tr in zip(tick_seconds.astype(int), tick_trs):
        if seconds % 60 == 0:
            major_ticks.append(tr)
            major_labels.append(str(seconds))
        else:
            minor_ticks.append(tr)
    draw_intervals(ax, responsive, participant, run, 0, COLORS["responsive"], 0.2)
    draw_intervals(ax, nonresponsive, participant, run, 0, COLORS["non-responsive"], 0.17)
    ax.vlines(presses, 0.42, 0.58, color="#222222", linewidth=0.7, alpha=0.95, zorder=3)
    run_segments = metadata[(metadata["participant"] == participant) & (metadata["run"] == run)]
    for row in run_segments.itertuples(index=False):
        ax.add_patch(Rectangle((row.segment_start_TR, 0.08),
                     row.segment_end_TR_exclusive - row.segment_start_TR, 0.84,
                     facecolor="none", edgecolor=COLORS[row.label], linewidth=1.1, zorder=2))

    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xlim(0, total_trs)
    ax.set_xticks(major_ticks, major_labels)
    ax.set_xticks(minor_ticks, minor=True)
    ax.tick_params(axis="x", which="major", length=6, labelsize=8)
    ax.tick_params(axis="x", which="minor", length=3)
    ax.set_xlabel("Time within run (seconds)", labelpad=12, loc="left")
    ax.set_title(f"{participant} {run}: button presses, labeled intervals, and motion-free segments", loc="left")
    legend = [Patch(facecolor=COLORS["responsive"], alpha=0.2, label="Responsive interval"),
              Patch(facecolor=COLORS["non-responsive"], alpha=0.17, label="Non-responsive interval"),
              Line2D([0], [0], color="#222222", marker="|", linestyle="none", markersize=10,
                     label="Button press"),
              Patch(facecolor="none", edgecolor="0.25", linewidth=1.1, label="Segment")]
    ax.legend(handles=legend, loc="upper left", bbox_to_anchor=(0, -0.43),
              borderaxespad=0, ncol=4, frameon=True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    output_path = output_dir / f"{participant}_{run}_labels.png"
    fig.subplots_adjust(left=min(0.08, 0.8 / width), right=0.995, top=0.84, bottom=0.42)
    fig.savefig(output_path, dpi=dpi, facecolor="white")
    plt.close(fig)
    return output_path


def main():
    args = parse_args()
    if args.pixels_per_tr <= 0 or args.dpi <= 0:
        raise ValueError("--pixels-per-tr and --dpi must be positive")
    metadata, responsive, nonresponsive = load_inputs(args.segment_dir)
    output_dir = args.segment_dir / "labels"
    output_dir.mkdir(parents=True, exist_ok=True)
    participants = sorted(metadata["participant"].dropna().astype(str).unique(), key=numeric_key)
    for participant in participants:
        (output_dir / f"{participant}_labels.png").unlink(missing_ok=True)
    outputs = [plot_run(participant, run_data, metadata, responsive, nonresponsive, output_dir,
                        args.pixels_per_tr, args.dpi)
               for participant in participants for run_data in participant_runs(participant)]
    outputs = [path for path in outputs if path is not None]
    print(f"Wrote {len(outputs)} participant-run PNGs to {output_dir}")


if __name__ == "__main__":
    main()
