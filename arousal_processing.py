import numpy as np


def load_arousals(bps, seconds_between_bps=30):
    """
    Return a list of times (in seconds, relative to the first TR timestamp) when arousals occured
    Arousals are defined as the timestamp of the 1st button press after at least `seconds_between_bps` of no button presses.
    The first button press is not considered arousal; we only consider a button press to be an arousal if it is preceded by a gap of at least `seconds_between_bps` since the previous button press.
    """
    interclick_intervals = np.diff(bps)
    arousal_indices = np.where(interclick_intervals >= seconds_between_bps)[0] + 1 # +1 to get the index of the button press after the long gap
    arousal_times = bps[arousal_indices]
    return arousal_times

def load_sleep_onsets(bps, seconds_between_bps=30):
    """
    Return a list of times (in seconds, relative to the first TR timestamp) when sleep onset events occurred.

    Sleep onsets are defined as the timestamp of the last button press before at least `seconds_between_bps` seconds of no button presses.

    For a gap between bps[i] and bps[i+1] that is >= `seconds_between_bps`, the sleep onset is taken to be bps[i].
    """
    interclick_intervals = np.diff(bps)
    sleep_onset_indices = np.where(interclick_intervals >= seconds_between_bps)[0]
    sleep_onset_times = bps[sleep_onset_indices]
    return sleep_onset_times

def load_nearest_following_TR(list_of_timestamps, TR=0.98):
    """
    Return a list of TR indices corresponding to the nearest TR timestamp
    following each timestamp in the input list.
    """
    TRs = np.ceil(list_of_timestamps / TR).astype(int)
    return TRs

def load_no_bp_intervals(bps, seconds_between_bps=30):
    """
    Return intervals of no button presses lasting at least `seconds_between_bps`.

    Sleep onset = timestamp of last BP before long no-BP gap
    Arousal = timestamp of first BP after long no-BP gap

    Returns:
    - intervals: list of tuples (sleep_onset_time, arousal_time)
    """
    bps = np.asarray(bps)
    interclick_intervals = np.diff(bps)
    long_gap_indices = np.where(interclick_intervals >= seconds_between_bps)[0]
    sleep_onset_times = bps[long_gap_indices]
    arousal_times = bps[long_gap_indices + 1]
    intervals = list(zip(sleep_onset_times, arousal_times))
    return intervals
