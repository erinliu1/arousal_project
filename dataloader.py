import os
import numpy as np
import pandas as pd
import joblib

from all_rois import all_cortical_rois, all_thalamic_rois, get_all_rois_with_cortical_and_thalamic

all_rois = get_all_rois_with_cortical_and_thalamic()
print(len(all_rois))
class MyDataLoader:
    def __init__(self, roi, bootstrap_seed):
        # roi can be 'all' or any ROI in all_rois
        # if 'cortex', then use all cortical rois; if 'wholeThalamus', then use all thalamic rois

        self.all_rois = ['LC', 'DR', 'MnR', 'mRt', 'PAG', 'PnO', 'PTg', 'VTA', 'BasalForebrain', 'POA', 'LH', 'caudalanteriorcingulate', 'caudalmiddlefrontal', 'cuneus', 'entorhinal', 'frontalpole', 'fusiform', 'insula', 'isthmuscingulate', 'lateraloccipital', 'lateralorbitofrontal', 'lingual', 'medialorbitofrontal', 'ostralmiddlefrontal', 'paracentral', 'parahippocampal', 'pericalcarine', 'postcentral', 'posteriorcingulate', 'precentral', 'precuneus', 'rostralanteriorcingulate', 'superiorfrontal', 'superiorparietal', 'temporalpole', 'AV', 'CM', 'LD', 'LGN', 'MD', 'PUL', 'VA', 'VLa', 'VLp', 'VPL']

        self.bootstrap_seed = bootstrap_seed
        np.random.seed(self.bootstrap_seed)

        self.all_training_data = joblib.load(f'/orcd/data/ldlewis/001/om/erinliu/Arousal_Project/all_training_data.joblib')

        if roi == 'cortex':
            self.my_rois = ['caudalanteriorcingulate', 'caudalmiddlefrontal', 'cuneus', 'entorhinal', 'frontalpole', 'fusiform', 'insula', 'isthmuscingulate', 'lateraloccipital', 'lateralorbitofrontal', 'lingual', 'medialorbitofrontal', 'ostralmiddlefrontal', 'paracentral', 'parahippocampal', 'pericalcarine', 'postcentral', 'posteriorcingulate', 'precentral', 'precuneus', 'rostralanteriorcingulate', 'superiorfrontal', 'superiorparietal', 'temporalpole']

        elif roi == 'wholeThalamus':
            self.my_rois = ['AV', 'CM', 'LD', 'LGN', 'MD', 'PUL', 'VA', 'VLa', 'VLp', 'VPL']

        elif roi == 'all':
            self.my_rois = self.all_rois

        elif roi in ['LC', 'DR', 'MnR', 'mRt', 'PAG', 'PnO', 'PTg', 'VTA', 'BasalForebrain', 'POA', 'LH']:
            self.my_rois = [roi]
        
        else:
            raise ValueError(f"ROI {roi} not found.")

        self.roi_indices = [self.all_rois.index(roi) for roi in self.my_rois]

        # self.dataset, self.loocv_lookup = self.construct_dataset()

    def get_all_participants(self):
        return sorted(list(self.all_training_data.keys()))
    
    def get_segments(self, participant, segment_type):
        segments = (
            self.all_training_data
            .get(participant, {})
            .get(segment_type, None)
        )

        if segments is not None:
            return segments[:, self.roi_indices, :]

        return None
    
    def construct_dataset(self):
        data = {}
        loocv_lookup = {}
        i = 0

        for participant in self.get_all_participants():
            prearousal = self.get_segments(participant, 'prearousal')
            sleep = self.get_segments(participant, 'sleep')
            wake = self.get_segments(participant, 'wake')
            sleep_onset = self.get_segments(participant, 'sleep_onset')

            if prearousal is None or sleep is None or wake is None:
                continue

                n_prearousal = prearousal.shape[0]
                n_sleep = sleep.shape[0]
                n_wake = wake.shape[0]

                n_samples = min(n_prearousal, n_sleep, n_wake)

                prearousal = prearousal[np.random.choice(n_prearousal, n_samples, replace=False)]
                sleep = sleep[np.random.choice(n_sleep, n_samples, replace=False)]
                wake = wake[np.random.choice(n_wake, n_samples, replace=False)]
                
                labels = (
                    [1] * n_samples +
                    [0] * n_samples +
                    [2] * n_samples
                )

                null_labels = np.random.permutation(labels)

                if participant not in data:
                    data[participant] = {}

                data[participant][run_id] = {
                    'prearousal': prearousal,
                    'sleep': sleep,
                    'wake': wake,
                    'true': labels,
                    'null': null_labels
                }

                loocv_lookup[i] = (participant, run_id)
                i += 1

        return data, loocv_lookup

dataloader = MyDataLoader(roi='all', bootstrap_seed=0)
# dataset, loocv_lookup = dataloader.dataset, dataloader.loocv_lookup
# for participant, participant_data in dataset.items():
#     for run_id, run_data in participant_data.items():
#         print(
#             participant,
#             run_id,
#             run_data['prearousal'].shape,
#             run_data['sleep'].shape,
#             run_data['wake'].shape,
#             len(run_data['true']),
#             len(run_data['null'])
#         )
#         break
#     break