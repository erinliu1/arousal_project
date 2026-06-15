import os
import re
import numpy as np

def framewise_displacement(participant, run):
    base_dir = '/orcd/data/ldlewis/001/om/erinliu/Arousal_Project/nicks_data'
    if type(run) == str:
        run = int(re.match(r'run(.*)', run).group(1))
    mclogs = os.path.join(base_dir, participant, f'run{run:02d}', 'mclog.mcdat')
    
    # load mclogs file - matrix where each row is one fMRI volume (time point) and each column is a different motion parameter
    try:
        a = np.loadtxt(mclogs)
    except Exception as e:
        print(mclogs)
        raise ValueError(e)
    
    rotations = a[:, 1:4]     # rotations (degrees): pitch, roll, yaw
    translations = a[:, 4:7]  # translations (mm): x, y, z

    # convert rotations from degrees radians, and then to mm by multiplying by 50 (approximate radius of the brain in mm)
    # after this step, all 6 motion parameters are in mm
    rotations_rad = np.deg2rad(rotations)
    rotations_mm = rotations_rad * 50

    # compute frame-to-frame differences for each motion parameter; i.e. how much did the head move since the previous volume
    all_params = np.hstack([rotations_mm, translations])
    differences = np.diff(all_params, axis=0) # differences between consecutive rows; shape = (num_frames - 1, 6)

    # framewise displacement value --> take the absolute value of each motion change and sum across all 6 params
    framewise_displacement = np.sum(np.abs(differences), axis=1)  # shape = (num_frames - 1, )
    return framewise_displacement

def get_motion_TRs(framewise_displacement, threshold=0.5, surrounding_TRs=2):
    # compute the indices where framewise displacement exceeds the threshold (default 0.5 mm)
    # include +/- surrounding_TRs around each high-motion TR and return list of unique TR indices
    high_motion_TRs = np.where(framewise_displacement > threshold)[0] + 1  # +1 because fd[i] is the movement from TR i to TR i+1
    all_motion_TRs = set()
    for tr in high_motion_TRs:
        for offset in range(-surrounding_TRs, surrounding_TRs + 1):
            tr_candidate = tr + offset
            if tr_candidate >= 0 and tr_candidate < len(framewise_displacement) + 1:
                all_motion_TRs.add(int(tr_candidate))
    return sorted(all_motion_TRs) # TR indices