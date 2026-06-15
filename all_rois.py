all_rois = ['LC', 'DR', 'MnR', 'mRt', 'PAG', 'PnO', 'PTg', 'VTA', 'BasalForebrain', 'POA', 'LH', 'wholeThalamus', 'cortex']


all_cortical_rois = ['caudalanteriorcingulate', 'caudalmiddlefrontal', 'cuneus', 'entorhinal', 'frontalpole', 'fusiform', 'insula', 'isthmuscingulate', 'lateraloccipital', 'lateralorbitofrontal', 'lingual', 'medialorbitofrontal', 'ostralmiddlefrontal', 'paracentral', 'parahippocampal', 'pericalcarine', 'postcentral', 'posteriorcingulate', 'precentral', 'precuneus', 'rostralanteriorcingulate', 'superiorfrontal', 'superiorparietal', 'temporalpole']


all_thalamic_rois = ['AV', 'CM', 'LD', 'LGN', 'MD', 'PUL', 'VA', 'VLa', 'VLp', 'VPL']

def get_all_rois_with_cortical_and_thalamic():
    all_rois_with_cortical_and_thalamic = all_rois + all_cortical_rois + all_thalamic_rois
    all_rois_with_cortical_and_thalamic.remove('cortex')
    all_rois_with_cortical_and_thalamic.remove('wholeThalamus')
    return all_rois_with_cortical_and_thalamic