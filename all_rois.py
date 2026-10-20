aan_rois = ['LC', 'DR', 'MnR', 'mRt', 'PAG', 'PnO', 'PTg', 'VTA', 'BasalForebrain', 'POA', 'LH']

cortical_rois = ['caudalanteriorcingulate', 'caudalmiddlefrontal', 'cuneus', 'entorhinal', 'frontalpole', 'fusiform', 'insula', 'isthmuscingulate', 'lateraloccipital', 'lateralorbitofrontal', 'lingual', 'medialorbitofrontal', 'rostralmiddlefrontal', 'paracentral', 'parahippocampal', 'pericalcarine', 'postcentral', 'posteriorcingulate', 'precentral', 'precuneus', 'rostralanteriorcingulate', 'superiorfrontal', 'superiorparietal', 'temporalpole']


thalamic_rois = ['AV', 'CM', 'LD', 'MD', 'PUL', 'VA', 'VLa', 'VLp', 'VPL']

motor_regions = ['postcentral', 'precentral']

all_rois = [roi for roi in (aan_rois + cortical_rois + thalamic_rois) if roi not in motor_regions]